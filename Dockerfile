ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.10-py3

FROM ${BASE_IMAGE}

ARG MEGATRON_DIR=/workspace/Megatron-Deepspeed

COPY requirements.txt ${MEGATRON_DIR}/

RUN pip install --upgrade pip && \
    pip install --no-build-isolation "flash-attn>=2.0.0" && \
    pip install sentencepiece && \
    # For make on data directory
    pip install pybind11 && \
    pip install -r ${MEGATRON_DIR}/requirements.txt

COPY . ${MEGATRON_DIR}/

RUN cd ${MEGATRON_DIR}/megatron/data && make
RUN pip install -e ${MEGATRON_DIR}

WORKDIR ${MEGATRON_DIR}