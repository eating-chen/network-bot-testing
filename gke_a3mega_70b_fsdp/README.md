# Llama 3.1 70B QLoRA + FSDP on GKE A3 Mega

This variant runs continual pre-training on two `a3-megagpu-8g` nodes: 16 H100
80GB GPUs in total. It reuses the dependency image built from
`gke_a3mega/Dockerfile`; Python and Accelerate configuration are mounted from a
ConfigMap, so code edits do not require rebuilding the image.

## Important differences from the 8B version

- `meta-llama/Llama-3.1-70B-Instruct`
- FSDP1 `FULL_SHARD` through Accelerate
- QLoRA NF4 with BF16 compute **and BF16 quant storage**
- `fsdp_use_orig_params: false` plus PEFT-aware auto wrapping
- CPU-efficient model loading, without CPU parameter offload
- sharded training checkpoints and a final portable PEFT adapter save
- fused AdamW; do not change this to `paged_adamw_8bit`, which has a known
  FSDP+QLoRA checkpoint failure

The default LoRA plan remains rank 128, alpha 256, dropout 0.05. At rank 128,
70B has roughly 1.66B adapter parameters, so this is a high-capacity CPT setup.
If that is more capacity than intended, set `LORA_R=64` and `LORA_ALPHA=128`.

The default effective batch is `1 x 16 x 8 = 128` sequences per optimizer
step. With the previously measured approximately 9.62B train tokens and context
4096, that is approximately 18,350 optimizer steps per epoch, or 36,700 for two
epochs.

## Prepare and launch

Complete the prerequisites in `gke_a3mega/README.md`, build its dependency
image once, and replace every `YOUR_*` value in `jobset.yaml`.

Create or update the code ConfigMap:

```bash
bash gke_a3mega_70b_fsdp/sync_code.sh
```

Launch and follow logs:

```bash
kubectl apply -f gke_a3mega_70b_fsdp/jobset.yaml
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=cpt-a3mega-70b -w
kubectl logs -l jobset.sigs.k8s.io/jobset-name=cpt-a3mega-70b -c trainer --prefix -f
```

After changing Python or FSDP configuration, recreate the JobSet. It resumes
from the latest shared checkpoint by default:

```bash
kubectl delete jobset cpt-a3mega-70b
bash gke_a3mega_70b_fsdp/sync_code.sh
kubectl apply -f gke_a3mega_70b_fsdp/jobset.yaml
```

## First-run recommendation

Before the full run, add these environment variables to the trainer container:

```yaml
- name: MAX_TRAIN_SAMPLES
  value: "256"
- name: MAX_EVAL_SAMPLES
  value: "64"
- name: MAX_STEPS
  value: "5"
- name: RUN_TEST_AFTER_TRAINING
  value: "false"
```

Verify that all 16 ranks start, the NCCL log selects the TCPXO/FasTrak plugin,
FSDP is enabled, loss is finite, and checkpoints can be written and resumed.
Then remove the smoke-test overrides.

The final artifact is a PEFT adapter, not a standalone merged 70B model. Load it
with the original Llama base model for inference. Merge outside the FSDP
training job only if a standalone merged model is required.
