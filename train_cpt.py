"""Trace-first continual pre-training (CPT) with TRL 1.12.0.

Colab install (restart the runtime after installation if Colab asks you to):
    !pip install -U "trl[peft,quantization]==1.12.0"
    !pip install -U "kernels>=0.11.0"

The default is a two-epoch Llama 3.1 8B Instruct QLoRA CPT run. Accept its
Hugging Face access terms, mount Google Drive, and log in first. The processed
Parquet splits are read from /content/drive/MyDrive/network/CPT_data. Then run:
    python train_cpt.py

Set both sample limits to None and max_steps=-1 for the complete
num_train_epochs run. For a larger model on a small GPU, use "lora" or
"qlora".
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

LOGGER = logging.getLogger("cpt")
EXPECTED_TRL = Version("1.12.0")


@dataclass(frozen=True)
class TrainConfig:
    # Gated model: accept its HF terms and authenticate before downloading.
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    data_dir: str = "/content/drive/MyDrive/network/CPT_data"
    output_dir: str = (
        "/content/drive/MyDrive/network/models/network-cpt-llama-3.1-8b-instruct-qlora"
    )

    # QLoRA is the Colab-friendly plan; "full" would require a much larger GPU.
    train_mode: str = "qlora"  # full | lora | qlora
    max_length: int = 4096
    packing_strategy: str = "bfd_split"  # preserves every token in long documents

    learning_rate: float = 1e-4  # conservative QLoRA CPT starting point
    num_train_epochs: float = 2.0
    max_steps: int = -1  # negative means train for num_train_epochs
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    logging_steps: int = 1
    eval_steps: int = 5000
    save_steps: int = 5000
    save_total_limit: int = 3
    dataset_num_proc: int = 2
    dataloader_num_workers: int = 0  # 0 gives clearer Colab stack traces
    seed: int = 42

    # Optional deterministic subsets for an even faster debugging pass.
    max_train_samples: int | None = None  # use None for the complete corpus
    max_eval_samples: int | None = None
    run_test_after_training: bool = True
    resume_from_checkpoint: bool = True

    # Precompiled Hub kernel: avoids building flash-attn from source in Colab.
    attn_implementation: str = "kernels-community/flash-attn2"
    trust_remote_code: bool = False

    # PEFT settings, used only in lora/qlora mode.
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05


CONFIG = TrainConfig()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def validate_config(cfg: TrainConfig) -> None:
    if cfg.train_mode not in {"full", "lora", "qlora"}:
        raise ValueError("train_mode must be one of: full, lora, qlora")
    if cfg.packing_strategy not in {"wrapped", "bfd_split"}:
        raise ValueError("packing_strategy must be wrapped or bfd_split")
    flash_attention_backends = {
        "flash_attention_2",
        "flash_attention_3",
        "kernels-community/flash-attn2",
        "kernels-community/flash-attn3",
    }
    if (
        cfg.packing_strategy == "bfd_split"
        and cfg.attn_implementation not in flash_attention_backends
    ):
        raise ValueError(
            "TRL bfd_split uses padding-free batches and needs FlashAttention. "
            "Use packing_strategy='wrapped', or select a supported FlashAttention backend."
        )
    if cfg.max_length < 2:
        raise ValueError("max_length must be at least 2")
    if cfg.max_steps == 0:
        raise ValueError("max_steps must be -1 (full epochs) or a positive integer")


def log_environment() -> None:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    LOGGER.info("Python=%s | platform=%s", platform.python_version(), platform.platform())
    LOGGER.info(
        "torch=%s | transformers=%s | datasets=%s | trl=%s",
        torch.__version__,
        transformers.__version__,
        datasets.__version__,
        trl.__version__,
    )
    LOGGER.info("CUDA available=%s | GPU=%s", torch.cuda.is_available(), cuda_name)
    if Version(trl.__version__) != EXPECTED_TRL:
        LOGGER.warning(
            "This script targets TRL %s, but the runtime has %s. "
            "Install the pinned Colab dependency shown at the top of this file.",
            EXPECTED_TRL,
            trl.__version__,
        )


def resolve_data_files(data_dir: Path) -> dict[str, str]:
    files = {split: data_dir / f"{split}.parquet" for split in ("train", "validation", "test")}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing processed CPT split(s):\n  "
            + "\n  ".join(missing)
            + "\nRun the preprocessing pipeline first or change CONFIG.data_dir."
        )
    return {split: str(path) for split, path in files.items()}


def take_subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("Sample limits must be positive or None")
    return dataset.shuffle(seed=seed).select(range(limit))


def load_corpus(cfg: TrainConfig) -> DatasetDict:
    data_files = resolve_data_files(Path(cfg.data_dir))
    corpus = load_dataset("parquet", data_files=data_files)
    required = {"text"}
    for split, dataset in corpus.items():
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(f"{split} is missing columns: {sorted(missing)}")
        LOGGER.info("raw %-10s rows=%d columns=%s", split, len(dataset), dataset.column_names)

    corpus["train"] = take_subset(corpus["train"], cfg.max_train_samples, cfg.seed)
    corpus["validation"] = take_subset(corpus["validation"], cfg.max_eval_samples, cfg.seed)
    corpus["test"] = take_subset(corpus["test"], cfg.max_eval_samples, cfg.seed)
    for split, dataset in corpus.items():
        if len(dataset) == 0:
            raise ValueError(f"{split} split is empty")
        if "estimated_tokens" in dataset.column_names:
            estimated_tokens = sum(int(value or 0) for value in dataset["estimated_tokens"])
        else:
            estimated_tokens = sum(len(text) // 4 for text in dataset["text"])
        LOGGER.info(
            "selected %-10s rows=%d | preprocessing estimate=%s tokens",
            split,
            len(dataset),
            f"{estimated_tokens:,}",
        )
    return corpus


def load_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("CPT requires a tokenizer with an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        LOGGER.info("Tokenizer has no PAD token; using EOS as PAD (no vocabulary resize).")
    tokenizer.padding_side = "right"
    LOGGER.info(
        "tokenizer=%s | vocab=%d | BOS=%s | EOS=%s | PAD=%s",
        tokenizer.__class__.__name__,
        len(tokenizer),
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    return tokenizer


def choose_precision() -> tuple[torch.dtype, bool, bool]:
    if not torch.cuda.is_available():
        return torch.float32, False, False
    bf16 = torch.cuda.is_bf16_supported()
    return (torch.bfloat16 if bf16 else torch.float16), bf16, not bf16


def build_model_and_peft(cfg: TrainConfig):
    dtype, _, _ = choose_precision()
    quantization_config = None
    if cfg.train_mode == "qlora":
        from transformers import BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU and bitsandbytes")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        dtype=dtype,
        attn_implementation=cfg.attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.config.use_cache = False

    context = getattr(model.config, "max_position_embeddings", None)
    if context is not None and cfg.max_length > context:
        raise ValueError(f"max_length={cfg.max_length} exceeds model context={context}")

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


def parameter_stats(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


class TraceCallback(TrainerCallback):
    """Print compact step metrics and keep the same records in trace.jsonl."""

    def __init__(self, trace_file: Path) -> None:
        self.trace_file = trace_file

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero or not logs:
            return
        record: dict[str, Any] = {"step": state.global_step}
        for key, value in logs.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                record[key] = value
        if torch.cuda.is_available():
            record["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
            record["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        LOGGER.info("TRACE %s", json.dumps(record, ensure_ascii=False, sort_keys=True))
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_training_args(cfg: TrainConfig, bf16: bool, fp16: bool) -> SFTConfig:
    # chunked_nll is TRL 1.12's lower-memory equivalent of NLL. PEFT with
    # all-linear may wrap lm_head, so LoRA modes use ordinary NLL.
    loss_type = "chunked_nll" if cfg.train_mode == "full" else "nll"
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
        loss_type=loss_type,
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
        tf32=bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
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
        seed=cfg.seed,
        data_seed=cfg.seed,
        remove_unused_columns=True,
    )


def trace_dataset(trainer: SFTTrainer, tokenizer) -> None:
    train = trainer.train_dataset
    LOGGER.info("prepared train rows=%d | packing=%s", len(train), trainer.args.packing_strategy)
    sample = train[0]
    ids = sample["input_ids"]
    labels = sample.get("labels", ids)
    trained = sum(label != -100 for label in labels)
    LOGGER.info("prepared sample tokens=%d | loss tokens=%d", len(ids), trained)
    LOGGER.info("decoded sample start:\n%s", tokenizer.decode(ids[:256]))

    batch = next(iter(trainer.get_train_dataloader()))
    shapes = {key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
    LOGGER.info("first batch tensor shapes=%s", shapes)
    if "labels" in batch:
        active = int((batch["labels"] != -100).sum().item())
        LOGGER.info("first batch active loss tokens=%d", active)


def find_resume_checkpoint(cfg: TrainConfig) -> str | None:
    output_dir = Path(cfg.output_dir)
    if not cfg.resume_from_checkpoint or not output_dir.is_dir():
        return None
    checkpoint = get_last_checkpoint(str(output_dir))
    if checkpoint:
        LOGGER.info("Resuming from %s", checkpoint)
    return checkpoint


def save_run_config(cfg: TrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    setup_logging()
    validate_config(CONFIG)
    log_environment()
    set_seed(CONFIG.seed)
    save_run_config(CONFIG)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    corpus = load_corpus(CONFIG)
    tokenizer = load_tokenizer(CONFIG)
    model, peft_config = build_model_and_peft(CONFIG)
    _, bf16, fp16 = choose_precision()
    args = build_training_args(CONFIG, bf16, fp16)
    trace_file = Path(CONFIG.output_dir) / "trace.jsonl"

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=corpus["train"],
        eval_dataset=corpus["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[TraceCallback(trace_file)],
    )
    total, trainable = parameter_stats(trainer.model)
    LOGGER.info(
        "parameters total=%s | trainable=%s (%.3f%%)",
        f"{total:,}",
        f"{trainable:,}",
        100 * trainable / total,
    )
    trace_dataset(trainer, tokenizer)

    result = trainer.train(resume_from_checkpoint=find_resume_checkpoint(CONFIG))
    trainer.save_model(CONFIG.output_dir)
    tokenizer.save_pretrained(CONFIG.output_dir)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    if CONFIG.run_test_after_training:
        test_metrics = trainer.evaluate(corpus["test"], metric_key_prefix="test")
        test_loss = test_metrics.get("test_loss")
        if test_loss is not None:
            test_metrics["test_perplexity"] = (
                math.exp(test_loss) if test_loss < 100 else float("inf")
            )
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    LOGGER.info("Done. Model, checkpoints, metrics, config, and trace are in %s", CONFIG.output_dir)


if __name__ == "__main__":
    # Avoid tokenizer worker warnings after datasets multiprocessing.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
