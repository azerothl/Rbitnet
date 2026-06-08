#!/usr/bin/env python3
"""
Minimal LoRA + SFT recipe for Llama-style causal LMs (Transformers + TRL + PEFT).

Output: merged or adapter weights under --output-dir (Safetensors). Convert to GGUF
with llama.cpp separately; see training/README.md and docs/TRAINING_AND_COMPATIBILITY.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0).",
    )
    p.add_argument(
        "--dataset-jsonl",
        required=True,
        type=Path,
        help="JSONL with `instruction` and `output` string fields per line.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for checkpoints / final adapter or merged weights.",
    )
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--merge-and-save",
        action="store_true",
        help="Merge LoRA into base weights and save full model (reloads base on CPU; needs significant CPU RAM).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 where supported (recommended on Ampere+).",
    )
    p.add_argument(
        "--int4",
        action="store_true",
        help="Load base model in 4-bit (QLoRA-style); requires bitsandbytes installed.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Pass trust_remote_code=True when loading the tokenizer and model. "
            "Only enable this for repositories whose custom code you trust."
        ),
    )
    return p.parse_args()


def formatting_func(tokenizer):
    def _fn(example: dict) -> str:
        instr = example.get("instruction", "")
        out = example.get("output", "")
        # Simple instruction template; replace with chat_template if you target chat models.
        return f"### Instruction:\n{instr}\n### Response:\n{out}{tokenizer.eos_token or ''}"

    return _fn


def main() -> int:
    args = parse_args()
    if not args.dataset_jsonl.is_file():
        print(f"error: dataset not found: {args.dataset_jsonl}", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if args.int4:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    load_kw: dict = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
    }
    if quant is not None:
        load_kw["quantization_config"] = quant
        load_kw["device_map"] = "auto"
    elif torch.cuda.is_available():
        load_kw["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kw)

    if args.int4:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    ds = load_dataset("json", data_files=str(args.dataset_jsonl), split="train")
    fmt = formatting_func(tokenizer)
    ds = ds.map(lambda x: {"text": fmt(x)})

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    )

    sft_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=5,
        save_steps=max(1, args.max_steps // 2),
        bf16=args.bf16,
        fp16=not args.bf16 and not args.int4,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        report_to="none",
        gradient_checkpointing=args.int4,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    adapter_dir = args.output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}")

    if args.merge_and_save:
        if args.int4:
            print(
                "error: --merge-and-save is not supported with --int4 in this minimal script.",
                file=sys.stderr,
            )
            return 1
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32),
            trust_remote_code=args.trust_remote_code,
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()
        merged_dir = args.output_dir / "merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged full weights to {merged_dir}")

    print("\nNext: convert Safetensors to GGUF with llama.cpp (see training/README.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
