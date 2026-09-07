"""Llama 3.1 8B Instruct CPT on two GKE A3 Mega nodes (16 H100 GPUs).

The Kubernetes manifest starts one process per GPU with torchrun. GPUDirect-TCPXO
is configured by the GKE NCCL installer, the TCPXO sidecar, and the shell launcher;
this Python process uses PyTorch's normal NCCL distributed backend.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
import trl
from datasets import Dataset, DatasetDict, load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

LOGGER = logging.getLogger("gke-cpt")
EXPECTED_TRL = Version("1.12.0")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none"}:
        return None
    return int(value)


@dataclass(frozen=True)
class TrainConfig:
    model_name_or_path: str = os.getenv("MODEL_NAME_OR_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    data_dir: str = os.getenv("CPT_DATA_DIR", "/mnt/cpt-data")
    output_dir: str = os.getenv("CPT_OUTPUT_DIR", "/mnt/cpt-output/llama-3.1-8b-instruct-qlora")

    train_mode: str = os.getenv("TRAIN_MODE", "qlora")  # full | lora | qlora
    max_length: int = int(os.getenv("MAX_LENGTH", "4096"))
    packing_strategy: str = os.getenv("PACKING_STRATEGY", "bfd_split")
    learning_rate: float = float(os.getenv("LEARNING_RATE", "1e-4"))
    num_train_epochs: float = float(os.getenv("NUM_TRAIN_EPOCHS", "2"))
    max_steps: int = int(os.getenv("MAX_STEPS", "-1"))

    # 4 sequences/GPU x 16 GPUs x 2 accumulation = global batch 128.
    per_device_train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "4"))
    per_device_eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "4"))
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACCUM_STEPS", "2"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.1"))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.03"))
    lr_scheduler_type: str = os.getenv("LR_SCHEDULER_TYPE", "cosine")
    max_grad_norm: float = float(os.getenv("MAX_GRAD_NORM", "1.0"))

    logging_steps: int = int(os.getenv("LOGGING_STEPS", "10"))
    eval_steps: int = int(os.getenv("EVAL_STEPS", "1000"))
    save_steps: int = int(os.getenv("SAVE_STEPS", "1000"))
    save_total_limit: int = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
    dataset_num_proc: int = int(os.getenv("DATASET_NUM_PROC", "16"))
    dataloader_num_workers: int = int(os.getenv("DATALOADER_NUM_WORKERS", "4"))
    seed: int = int(os.getenv("SEED", "42"))

    max_train_samples: int | None = env_optional_int("MAX_TRAIN_SAMPLES", None)
    # Full validation is expensive; override with "none" if it is intentional.
    max_eval_samples: int | None = env_optional_int("MAX_EVAL_SAMPLES", 8192)
    run_test_after_training: bool = env_bool("RUN_TEST_AFTER_TRAINING", True)
    resume_from_checkpoint: bool = env_bool("RESUME_FROM_CHECKPOINT", True)

    attn_implementation: str = os.getenv("ATTN_IMPLEMENTATION", "kernels-community/flash-attn2")
    trust_remote_code: bool = env_bool("TRUST_REMOTE_CODE", False)
    lora_r: int = int(os.getenv("LORA_R", "128"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "256"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))

    expected_world_size: int = int(os.getenv("EXPECTED_WORLD_SIZE", "16"))
    expected_local_world_size: int = int(os.getenv("EXPECTED_LOCAL_WORLD_SIZE", "8"))


CONFIG = TrainConfig()


def rank() -> int:
    return int(os.getenv("RANK", "0"))


def local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return rank() == 0


def setup_distributed(cfg: TrainConfig) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    allow_single = env_bool("ALLOW_SINGLE_PROCESS", False)
    if not allow_single and (
        world_size != cfg.expected_world_size or local_world_size != cfg.expected_local_world_size
    ):
        raise RuntimeError(
            "Expected torchrun with WORLD_SIZE="
            f"{cfg.expected_world_size} and LOCAL_WORLD_SIZE={cfg.expected_local_world_size}; "
            f"got {world_size} and {local_world_size}. Set ALLOW_SINGLE_PROCESS=1 "
            "only for a smoke test."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This GKE training entrypoint requires CUDA GPUs")
    if local_rank() >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank()} but only {torch.cuda.device_count()} GPUs are visible"
        )
    torch.cuda.set_device(local_rank())


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO if is_main_process() else logging.WARNING,
        format=f"%(asctime)s | rank={rank()} | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def validate_config(cfg: TrainConfig) -> None:
    if cfg.train_mode not in {"full", "lora", "qlora"}:
        raise ValueError("TRAIN_MODE must be full, lora, or qlora")
    if cfg.packing_strategy not in {"wrapped", "bfd_split"}:
        raise ValueError("PACKING_STRATEGY must be wrapped or bfd_split")
    flash_backends = {
        "flash_attention_2",
        "flash_attention_3",
        "kernels-community/flash-attn2",
        "kernels-community/flash-attn3",
    }
    if cfg.packing_strategy == "bfd_split" and cfg.attn_implementation not in flash_backends:
        raise ValueError("bfd_split requires a supported FlashAttention backend")
    if cfg.max_length < 2 or cfg.max_steps == 0:
        raise ValueError("MAX_LENGTH must be >=2 and MAX_STEPS must be -1 or positive")


def log_environment(cfg: TrainConfig) -> None:
    device = torch.cuda.current_device()
    LOGGER.info("Python=%s | platform=%s", platform.python_version(), platform.platform())
    LOGGER.info(
        "torch=%s | transformers=%s | datasets=%s | trl=%s",
        torch.__version__,
        transformers.__version__,
        datasets.__version__,
        trl.__version__,
    )
    LOGGER.info(
        "rank=%s/%s | local_rank=%s/%s | GPU=%s",
        rank(),
        os.getenv("WORLD_SIZE", "1"),
        local_rank(),
        os.getenv("LOCAL_WORLD_SIZE", "1"),
        torch.cuda.get_device_name(device),
    )
    global_batch = (
        cfg.per_device_train_batch_size
        * int(os.getenv("WORLD_SIZE", "1"))
        * cfg.gradient_accumulation_steps
    )
    LOGGER.info(
        "global train batch=%d sequences | tokens/update<=%s",
        global_batch,
        f"{global_batch * cfg.max_length:,}",
    )
    LOGGER.info(
        "TCPXO profile: NCCL_SOCKET_IFNAME=%s | NCCL_FASTRAK_IFNAME=%s | LD_LIBRARY_PATH=%s",
        os.getenv("NCCL_SOCKET_IFNAME", "unset"),
        os.getenv("NCCL_FASTRAK_IFNAME", "unset"),
        os.getenv("LD_LIBRARY_PATH", "unset"),
    )
    if Version(trl.__version__) != EXPECTED_TRL:
        LOGGER.warning("Script targets TRL %s; runtime has %s", EXPECTED_TRL, trl.__version__)


def resolve_data_files(data_dir: Path) -> dict[str, str]:
    files = {s: data_dir / f"{s}.parquet" for s in ("train", "validation", "test")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CPT splits:\n  " + "\n  ".join(missing))
    return {split: str(path) for split, path in files.items()}


def take_subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("Sample limits must be positive or none")
    return dataset.shuffle(seed=seed).select(range(limit))


def load_corpus(cfg: TrainConfig) -> DatasetDict:
    corpus = load_dataset("parquet", data_files=resolve_data_files(Path(cfg.data_dir)))
    for split, dataset in corpus.items():
        if "text" not in dataset.column_names:
            raise ValueError(f"{split} is missing the text column")
        LOGGER.info("raw %-10s rows=%d", split, len(dataset))
    corpus["train"] = take_subset(corpus["train"], cfg.max_train_samples, cfg.seed)
    corpus["validation"] = take_subset(corpus["validation"], cfg.max_eval_samples, cfg.seed)
    corpus["test"] = take_subset(corpus["test"], cfg.max_eval_samples, cfg.seed)
    for split, dataset in corpus.items():
        if len(dataset) == 0:
            raise ValueError(f"{split} split is empty")
        LOGGER.info("selected %-10s rows=%d", split, len(dataset))
    if is_main_process() and "estimated_tokens" in corpus["train"].column_names:
        tokens = sum(int(value or 0) for value in corpus["train"]["estimated_tokens"])
        global_batch = (
            cfg.per_device_train_batch_size
            * int(os.getenv("WORLD_SIZE", "1"))
            * cfg.gradient_accumulation_steps
        )
        LOGGER.info(
            "estimated train tokens=%s | optimizer steps/epoch≈%s",
            f"{tokens:,}",
            f"{math.ceil(tokens / (cfg.max_length * global_batch)):,}",
        )
    return corpus


def load_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code, use_fast=True
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("CPT tokenizer needs an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def choose_precision() -> tuple[torch.dtype, bool, bool]:
    bf16 = torch.cuda.is_bf16_supported()
    return (torch.bfloat16 if bf16 else torch.float16), bf16, not bf16


def build_model_and_peft(cfg: TrainConfig):
    dtype, _, _ = choose_precision()
    quantization_config = None
    model_kwargs: dict[str, Any] = {}
    if cfg.train_mode == "qlora":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": local_rank()}
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=cfg.trust_remote_code,
        **model_kwargs,
    )
    model.config.use_cache = False
    context = getattr(model.config, "max_position_embeddings", None)
    if context is not None and cfg.max_length > context:
        raise ValueError(f"MAX_LENGTH={cfg.max_length} exceeds model context={context}")

    peft_config = None
    if cfg.train_mode in {"lora", "qlora"}:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    return model, peft_config


class TraceCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero or not logs:
            return
        record: dict[str, Any] = {"step": state.global_step}
        record.update(
            {k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool)) or v is None}
        )
        record["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        record["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        LOGGER.info("TRACE %s", json.dumps(record, ensure_ascii=False, sort_keys=True))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def training_args(cfg: TrainConfig, bf16: bool, fp16: bool) -> SFTConfig:
    return SFTConfig(
        output_dir=cfg.output_dir,
        run_name=Path(cfg.output_dir).name,
        max_length=cfg.max_length,
        packing=True,
        packing_strategy=cfg.packing_strategy,
        eval_packing=True,
        dataset_text_field="text",
        dataset_num_proc=cfg.dataset_num_proc,
        shuffle_dataset=True,
        completion_only_loss=False,
        loss_type="chunked_nll",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=fp16,
        tf32=True,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        report_to="none",
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.dataloader_num_workers > 0,
        ddp_find_unused_parameters=False,
        save_on_each_node=False,
        log_on_each_node=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=True,
    )


def save_run_config(cfg: TrainConfig) -> None:
    if not is_main_process():
        return
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(cfg),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "trl": trl.__version__,
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def resume_checkpoint(cfg: TrainConfig) -> str | None:
    if not cfg.resume_from_checkpoint or not Path(cfg.output_dir).is_dir():
        return None
    return get_last_checkpoint(cfg.output_dir)


def main() -> None:
    setup_distributed(CONFIG)
    setup_logging()
    validate_config(CONFIG)
    log_environment(CONFIG)
    set_seed(CONFIG.seed)
    save_run_config(CONFIG)
    torch.cuda.reset_peak_memory_stats()

    corpus = load_corpus(CONFIG)
    tokenizer = load_tokenizer(CONFIG)
    model, peft_config = build_model_and_peft(CONFIG)
    _, bf16, fp16 = choose_precision()
    trainer = SFTTrainer(
        model=model,
        args=training_args(CONFIG, bf16, fp16),
        train_dataset=corpus["train"],
        eval_dataset=corpus["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[TraceCallback(Path(CONFIG.output_dir) / "trace.jsonl")],
    )
    total = sum(p.numel() for p in trainer.model.parameters())
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    LOGGER.info(
        "parameters total=%s | trainable=%s (%.3f%%)",
        f"{total:,}",
        f"{trainable:,}",
        100 * trainable / total,
    )

    result = trainer.train(resume_from_checkpoint=resume_checkpoint(CONFIG))
    trainer.save_model(CONFIG.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(CONFIG.output_dir)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    if CONFIG.run_test_after_training:
        metrics = trainer.evaluate(corpus["test"], metric_key_prefix="test")
        if "test_loss" in metrics:
            metrics["test_perplexity"] = (
                math.exp(metrics["test_loss"]) if metrics["test_loss"] < 100 else float("inf")
            )
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
    LOGGER.info("Done: %s", CONFIG.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
