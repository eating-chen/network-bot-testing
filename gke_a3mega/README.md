# GKE A3 Mega CPT

This directory runs Llama 3.1 8B Instruct QLoRA CPT on two `a3-megagpu-8g`
nodes (16 H100 GPUs total). Each node runs one Pod and eight `torchrun`
processes. Inter-node collectives use NCCL through GPUDirect-TCPXO.

## Before submitting

1. Use an A3 Mega GKE cluster whose eight secondary networks are exposed as
   `vpc1` through `vpc8`. If your network names differ, update the Pod
   annotations in `jobset.yaml`.
2. Install the JobSet controller and the GKE TCPXO NCCL installer. The installer
   release, GPU driver, CUDA base image, NCCL plugin, and `tcpxo-daemon` must be
   a qualified combination. Do not install NCCL FastSocket for this workload.

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-tcpxo/nccl-tcpxo-installer.yaml
   ```

   Use the current JobSet installation command from the upstream installation
   guide rather than pinning an old controller manifest in this repository.
3. Enable the GCSFuse CSI driver. Upload `train.parquet`, `validation.parquet`,
   and `test.parquet` to `gs://YOUR_CPT_DATA_BUCKET/CPT_data/`.
4. Grant the Kubernetes service account access through Workload Identity to
   read the data bucket and write the output bucket.
5. Accept the Llama 3.1 access terms, then create the token Secret:

   ```bash
   kubectl create secret generic hf-token --from-literal=token="$HF_TOKEN"
   ```

## Build the dependency image

Choose a CUDA image compatible with the TCPXO release installed on the nodes.
The default Docker build argument is NVIDIA PyTorch 26.05 (CUDA 13.2), matching
the CUDA generation qualified by the TCPXO release used in this example.

```bash
docker build -t YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY/network-cpt:a3mega gke_a3mega
docker push YOUR_REGION-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPOSITORY/network-cpt:a3mega
```

The image contains dependencies only. `train_cpt.py` is mounted separately from
a ConfigMap, so changing training code does not require another Docker build.

## Sync code and launch

Replace all `YOUR_*` values in `jobset.yaml`. Before the first launch, and after
every change to `train_cpt.py`, update the ConfigMap:

```bash
bash gke_a3mega/sync_code.sh
```

Then launch the JobSet:

```bash
kubectl apply -f gke_a3mega/jobset.yaml
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=cpt-a3mega -w
kubectl logs -l jobset.sigs.k8s.io/jobset-name=cpt-a3mega -c trainer --prefix -f
```

Updating a ConfigMap cannot hot-reload Python that is already running. To start
a new run with the changed code, recreate the JobSet; it will resume from the
latest checkpoint in the output bucket by default:

```bash
kubectl delete jobset cpt-a3mega
bash gke_a3mega/sync_code.sh
kubectl apply -f gke_a3mega/jobset.yaml
```

Only rebuild and push the Docker image when `Dockerfile`, CUDA/PyTorch, TRL, or
another Python dependency changes.

The default effective batch is `4 × 16 × 2 = 128` sequences per optimizer
step, with 4096 tokens, `bfd_split`, QLoRA rank 128/alpha 256/dropout 0.05,
BF16, and `chunked_nll`. The full validation split is intentionally capped at
8192 raw rows. All settings can be overridden with the environment variables
defined in `train_cpt.py`.

The output bucket is shared by both nodes. Only global rank zero writes the run
config, tokenizer, trace, and checkpoints; all ranks participate in training.
The Hugging Face cache is node-local ephemeral storage, so each node downloads
and preprocesses once rather than using GCSFuse as a hot cache.
