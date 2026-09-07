ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Keep the CUDA/PyTorch stack from the NVIDIA image. The GKE TCPXO installer
# supplies its qualified NCCL core and network plugin at runtime.
RUN python -m pip install --no-cache-dir \
    "trl[peft,quantization]==1.12.0" \
    "kernels>=0.11.0" \
    "datasets>=4.7.0" \
    "transformers>=4.56.2"

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Training code is mounted from a ConfigMap. Rebuild this image only when the
# base CUDA image or Python dependencies change.
CMD ["bash"]
