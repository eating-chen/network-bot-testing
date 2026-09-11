ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

WORKDIR /workspace

# Keep the CUDA/PyTorch stack from the NVIDIA image. The GKE TCPXO installer
# supplies its qualified NCCL core and network plugin at runtime. Pin the
# Python training stack so rebuilding the image cannot cross major versions.
RUN python -m pip install --no-cache-dir \
    "trl==1.12.0" \
    "transformers==4.57.6" \
    "datasets==4.8.5" \
    "accelerate==1.14.0" \
    "peft==0.20.0" \
    "bitsandbytes==0.50.2" \
    "kernels==0.16.1" \
 && python -m pip check

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Training code is mounted from a ConfigMap. Rebuild this image only when the
# base CUDA image or Python dependencies change.
CMD ["bash"]
