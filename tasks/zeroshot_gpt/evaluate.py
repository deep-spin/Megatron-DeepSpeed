# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT zero-shot evaluation."""

import math
from functools import partial

import torch

import entmax

from megatron import get_args
from megatron import print_rank_0, is_last_rank
from megatron import get_tokenizer
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.checkpointing import load_checkpoint
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
from tasks.finetune_utils import build_data_loader
from deepspeed.accelerator import get_accelerator
from .datasets import build_dataset
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding

from megatron.optimizer import get_megatron_optimizer

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfig

import json

def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        config = core_transformer_config_from_args(get_args())

        if eval_metric in {"loss", "force_decoded_accuracy", "force_decoded_accuracy_at_k"}:
            parallel_output = True
        elif eval_metric == 'accuracy':
            parallel_output = False
        else:
            raise NotImplementedError('output type for {} evaluation metric '
                                      'is not supported.'.format(eval_metric))

        print_rank_0('building GPT model ...')

        args = get_args()
        config = core_transformer_config_from_args(args)
        if args.deepspeed:
            with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                    remote_device=None if args.remote_device == 'none' else args.remote_device,
                                    config_dict_or_path=args.deepspeed_config,
                                    enabled=args.zero_stage == 3,
                                    mpu=mpu):

                model = GPTModel(
                    config=config,
                    num_tokentypes=0,
                    parallel_output=True,
                    pre_process=pre_process,
                    post_process=post_process
                )
        else:
            model = GPTModel(config=config, num_tokentypes=0, parallel_output=parallel_output,
                         pre_process=pre_process, post_process=post_process)


        return model

    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().to(get_accelerator().device_name()).contiguous().byte()
    tokens_ = batch['text'].long().to(get_accelerator().device_name()).contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, attention_mask, position_ids, loss_mask

"""
if loss_function == "cross_entropy":
    # cross entropy
    # [b s] => [s b]
    labels = labels.transpose(0,1).contiguous()
    cross_entropy = sequence_parallel.vocab_sequence_parallel_cross_entropy if mpu.get_sequence_parallel_world_size() > 1 \
        else tensor_parallel.vocab_parallel_cross_entropy
    if fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = cross_entropy(output, labels)
    else:
        loss = cross_entropy(output.float(), labels)

    # [s b] => [b, s]
    loss = loss.transpose(0,1).contiguous()
    support = None
else:
    # now: the loss function is "entmax15", "sparsemax", or "entmax_bisect"
    loss_funcs = {
        "entmax15": partial(entmax.entmax15_loss, k=topk, return_support_size=True),
        "sparsemax": partial(entmax.sparsemax_loss, k=topk, return_support_size=True),
        "entmax_bisect": partial(entmax.entmax_bisect_loss, alpha=alpha, n_iter=n_iter)
    }
    f = loss_funcs[loss_function]
    b, s = labels.size()
    output = output.transpose(0, 1).contiguous()
    vocab_size = output.size(-1)
    if loss_function != "entmax_bisect":
        loss, support = f(output.float().view(-1, vocab_size), labels.view(-1))
    else:
        loss = f(output.float().view(-1, vocab_size), labels.view(-1))
        support = None
    loss = loss.view(b, s)

"""


def _compute_loss(output, labels, loss_mask, loss_function="cross_entropy", topk=512, alpha=1.5, n_iter=30):
    """
    Dimensions are confusing but I think I've figured it out. Based on
    process_batch (defined above) and forward_step, labels is [b s]. I assume
    loss_mask is the same shape. And I assume that output is [b s V] (it would
    be ridiculously confusing otherwise).

    Based on the documentation of tensor_parallel.vocab_parallel_cross_entropy,
    we can expect output (or rather, output[0], which should be the decoder
    output based on TransformerLanguageModel.forward) to be [s b V] and labels
    to be [s b].
    """
    print("output type before _compute_loss", type(output))
    if isinstance(output, torch.Tensor):
        print("size before indexing", output.size())
    output = output[0]  # based on how loss was previously computed
    print("size after indexing", output.size())

    if loss_function == "cross_entropy":
        # I believe (based on the commented-out block above) that this
        # function takes [s b] as its input.
        # But I'm not certain
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output.contiguous().float(), labels.contiguous())
    else:
        # now: the loss function is "entmax15", "sparsemax", or "entmax_bisect"
        loss_funcs = {
            "entmax15": partial(entmax.entmax15_loss, k=topk, return_support_size=True),
            "sparsemax": partial(entmax.sparsemax_loss, k=topk, return_support_size=True),
            "entmax_bisect": partial(entmax.entmax_bisect_loss, alpha=alpha, n_iter=n_iter)
        }
        f = loss_funcs[loss_function]
        print("labels size: ", labels.size())
        print("output size", output.size())
        vocab_size = output[0].size(-1)
        if loss_function != "entmax_bisect":
            losses, _ = f(output.float().view(-1, vocab_size), labels.view(-1))
        else:
            losses = f(output.float().view(-1, vocab_size), labels.view(-1))
        # losses = losses.view(b, s)

    loss = torch.sum(losses.view(-1) * loss_mask.contiguous().view(-1).float())

    return loss


def _force_decoded_accuracy(output, labels, loss_mask):
    """
    This is different from LAMBADA accuracy, which is only about getting the
    final word right based on a long context.

    Dimensions are confusing but I think I've figured it out. Based on
    process_batch (defined above) and forward_step, labels is [b s]. I assume
    loss_mask is the same shape. And I assume that output is [b s V] (it would
    be ridiculously confusing otherwise).
    """

    # the raw output is a tuple (same as for eval_metric=="loss")
    output = output[0]

    predictions = output.argmax(dim=-1).view(-1)
    correct = predictions.eq(labels.view(-1)).float()

    correct_sum = torch.sum(correct * loss_mask.contiguous().view(-1).float())
    return correct_sum


def _force_decoded_accuracy_at_k(output, labels, loss_mask, k):
    """
    Accuracy at k -- do any of the top k outputs match?

    This is different from LAMBADA accuracy, which is only about getting the
    final word right based on a long context.

    Dimensions are confusing but I think I've figured it out. Based on
    process_batch (defined above) and forward_step, labels is [b s]. I assume
    loss_mask is the same shape. And I assume that output is [b s V] (it would
    be ridiculously confusing otherwise).
    """

    # the raw output is a tuple (same as for eval_metric=="loss")
    output = output[0]

    _, predictions = torch.topk(output, k, dim=-1)
    predictions = predictions.view(-1, k)

    correct = predictions.eq(labels.view(-1, 1)).any(dim=-1).float()

    correct_sum = torch.sum(correct * loss_mask.contiguous().view(-1).float())
    return correct_sum


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    input_tensor = recv_forward()

    # Forward pass through the model.
    if not args.deepspeed:
        unwrapped_model = unwrap_model(
            model, (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if parallel_state.is_pipeline_last_stage():
        # For loss, return the unreduced loss.
        if eval_metric == 'loss':
            '''
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                output[0].contiguous().float(), labels.contiguous())
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            return loss
            '''

            loss = _compute_loss(
                output, labels, loss_mask,
                loss_function=args.loss_function, topk=args.entmax_topk, n_iter=args.entmax_n_iter, alpha=args.entmax_alpha
            )
            return loss

        if eval_metric == "force_decoded_accuracy":
            correct_sum = _force_decoded_accuracy(output, labels, loss_mask)
            return correct_sum

        if eval_metric == "force_decoded_accuracy_at_k":
            k = args.acc_k
            correct_sum = _force_decoded_accuracy_at_k(output, labels, loss_mask, k)
            return correct_sum

        # For accuracy, return the number of correctly predicted samples.
        if eval_metric == 'accuracy':
            outputs = torch.argmax(output, -1)
            correct = (outputs == labels).float()
            correct[(1 - loss_mask).bool()] = 1
            correct = correct.prod(-1)
            return correct.sum()

        raise NotImplementedError('forward method for evaluation metric {} '
                                  'is not implemented.'.format(eval_metric))
    return None


def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(output,
                                             group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = ' validation results on {} | '.format(task)
    if is_last_rank():
        if eval_metric == 'loss':
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += 'avg loss: {:.4E} | '.format(val_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)

            results = {
                "loss": val_loss.item(),
                "ppl": ppl,
                "ajusted_ppl": adjusted_ppl,
                "token_ratio": token_ratio
            }

            with open('./eval_results', 'w') as json_file:
                json.dump(results, json_file)

        elif eval_metric == 'accuracy':
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        elif eval_metric == "force_decoded_accuracy" or eval_metric == "force_decoded_accuracy_at_k":
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            acc = output / (num_tokenized_tokens - 1)
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total tokens: {:.4E} | '.format(num_tokenized_tokens)
            string += 'avg accuracy: {:.4E}'.format(acc)

            results = {
                "accuracy": acc.item(),
                "n_correct": output.item(),
                "n_tokens": num_tokenized_tokens
            }
            with open('./eval_results', 'w') as json_file:
                json.dump(results, json_file)

        else:
            raise NotImplementedError('evaluation method for {} metric is not '
                                      'implemented yet.'.format(eval_metric))

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)

def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.eval_metric is not None:
        eval_metric = args.eval_metric
    elif args.task == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.task == 'WIKITEXT103':
        eval_metric = 'loss'
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.task))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=False)

    if args.deepspeed:
        optimizer = None
        opt_param_scheduler = None
        model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                model_parameters=model[0].parameters(),
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                mpu=mpu if args.no_pipeline_parallel else None
            )
        model = [model]

    if args.load is not None:
        if args.task == "LAMBADA":
            _ = load_checkpoint(model, None, None, load_iteration=args.load_iteration, load_only_weights=True)
        else:
            _ = load_checkpoint(model, None, None, load_iteration=args.load_iteration)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Data stuff.
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)

    # Run evaluation.
    evaluate_and_print_results(args.task, dataloader, model, eval_metric)


    print_rank_0('done :-)')

