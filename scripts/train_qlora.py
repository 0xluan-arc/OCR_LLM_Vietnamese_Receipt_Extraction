"""Kịch bản huấn luyện tinh chỉnh tham số hiệu quả QLoRA (4-bit NF4) cho mô hình
Qwen/Qwen2.5-1.5B-Instruct hoặc Qwen/Qwen2.5-7B-Instruct theo cấp độ bộ nhớ tần số thấp
(C^(L) = L) của khung lý thuyết Nested Learning (NeurIPS 2025).

Hỗ trợ:
- Lượng tử hóa 4-bit NF4 với Double Quantization (bitsandbytes)
- LoRA Adapter (r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
- Huấn luyện qua SFTTrainer (trl) trên dữ liệu ChatML JSONL (qlora_train.jsonl & qlora_val.jsonl)
- Đánh giá tự động trên tập Test sau khi huấn luyện bằng receipt_extraction.evaluation.metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TARGET_LORA_MODULES: List[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_qlora_hyperparameters(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 2048,
    optimizer_type: str = "nested_m3",
    slow_freq_f: int = 4,
) -> Dict[str, Any]:
    """Trả về cấu hình chuẩn QLoRA (BitsAndBytes 4-bit NF4 + Peft LoraConfig + SFTConfig + Nested M3 Optimizer)."""
    return {
        "model_name": model_name,
        "quantization_config": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "lora_config": {
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(TARGET_LORA_MODULES),
        },
        "nested_optimizer_config": {
            "optimizer_type": optimizer_type,
            "slow_freq_f": slow_freq_f,
            "ns_iters": 5,
            "zeta": 0.25,
            "alpha": 0.5,
        },
        "training_args": {
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_seq_length": max_seq_length,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "logging_steps": 5,
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "fp16": True,
            "optim": "paged_adamw_8bit" if optimizer_type == "adamw" else "nested_m3_muon",
            "seed": 42,
        },
    }


def load_jsonl_dataset(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Đọc danh sách mẫu ChatML từ file JSONL."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu SFT tại: {jsonl_path}")
    items: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items


def run_qlora_training(
    model_name: str,
    train_file: Path,
    val_file: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 2e-4,
    optimizer_type: str = "nested_m3",
    slow_freq_f: int = 4,
    dry_run: bool = False,
) -> int:
    """Thực thi quy trình huấn luyện QLoRA bằng transformers, peft, bitsandbytes và trl."""
    hparams = build_qlora_hyperparameters(
        model_name=model_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        optimizer_type=optimizer_type,
        slow_freq_f=slow_freq_f,
    )

    train_samples = load_jsonl_dataset(train_file)
    val_samples = load_jsonl_dataset(val_file) if val_file.exists() else []

    print("=" * 70)
    print("🚀 CẤU HÌNH HUẤN LUYỆN QLORA (NESTED LEARNING LOW-FREQUENCY LEVEL)")
    print("=" * 70)
    print(f"- Base Model        : {hparams['model_name']}")
    print(f"- Quantization      : 4-bit NF4, Double Quant={hparams['quantization_config']['bnb_4bit_use_double_quant']}")
    print(f"- LoRA Rank / Alpha : r={hparams['lora_config']['r']}, alpha={hparams['lora_config']['lora_alpha']}")
    print(f"- Target Modules    : {', '.join(hparams['lora_config']['target_modules'])}")
    print(f"- Dữ liệu Train/Val : {len(train_samples)} mẫu train / {len(val_samples)} mẫu val")
    print(f"- Thư mục Output    : {output_dir}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_out = output_dir / "qlora_training_config.json"
    config_out.write_text(json.dumps(hparams, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        print(f"✅ [DRY-RUN] Đã kiểm tra hợp lệ dữ liệu SFT và xuất cấu hình tại: {config_out}")
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as exc:
        print(
            f"⚠️ Thiếu thư viện huấn luyện GPU trên máy hiện tại ({exc}).\n"
            "💡 Để chạy huấn luyện trên Google Colab / Kaggle T4 GPU hoặc máy trạm có GPU NVIDIA, hãy cài đặt:\n"
            "   pip install -U torch transformers peft bitsandbytes trl datasets accelerate\n"
            "Hoặc mở trực tiếp notebook: notebooks/NCKH_Train_QLoRA_Qwen25.ipynb"
        )
        return 1

    if not torch.cuda.is_available():
        print(
            "⚠️ Không phát hiện GPU CUDA trên máy hiện tại (torch.cuda.is_available() == False).\n"
            "💡 Lượng tử hóa 4-bit NF4 (bitsandbytes) yêu cầu GPU NVIDIA (ví dụ: Tesla T4 trên Google Colab / Kaggle).\n"
            "👉 Hãy sử dụng notebook `notebooks/NCKH_Train_QLoRA_Qwen25.ipynb` để huấn luyện trên GPU."
        )
        return 1

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=hparams["lora_config"]["r"],
        lora_alpha=hparams["lora_config"]["lora_alpha"],
        lora_dropout=hparams["lora_config"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=hparams["lora_config"]["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def _format_chat(sample: Dict[str, Any]) -> Dict[str, str]:
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list([_format_chat(s) for s in train_samples])
    eval_ds = Dataset.from_list([_format_chat(s) for s in val_samples]) if val_samples else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=hparams["training_args"]["gradient_accumulation_steps"],
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        seed=42,
        report_to="none",
    )

    custom_optimizers = (None, None)
    if optimizer_type == "nested_m3":
        from receipt_extraction.optimizers.nested_optimizers import build_pytorch_m3_optimizer_class
        NestedM3Optimizer = build_pytorch_m3_optimizer_class()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        m3_opt = NestedM3Optimizer(trainable_params, lr=lr, slow_freq_f=slow_freq_f)
        custom_optimizers = (m3_opt, None)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=hparams["training_args"]["max_seq_length"],
        args=training_args,
        optimizers=custom_optimizers,
    )

    trainer.train()
    adapter_dir = output_dir / "final_lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"✅ Đã lưu trọng số LoRA Adapter hoàn chỉnh tại: {adapter_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Huấn luyện QLoRA 4-bit NF4 cho Qwen2.5-1.5B/7B-Instruct trên bài toán trích xuất hóa đơn Việt Nam."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tên mô hình trên Hugging Face (Qwen/Qwen2.5-1.5B-Instruct hoặc Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "qlora_train.jsonl"),
        help="File JSONL tập train định dạng ChatML.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "qlora_val.jsonl"),
        help="File JSONL tập validation định dạng ChatML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "qwen25_receipt_qlora"),
        help="Thư mục lưu checkpoint và LoRA adapter.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch huấn luyện (mặc định: 5).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size trên mỗi GPU (mặc định: 2).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (mặc định: 2e-4).")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["nested_m3", "adamw"],
        default="nested_m3",
        help="Bộ tối ưu hóa: nested_m3 (Multi-scale Momentum Muon, NeurIPS 2025) hoặc adamw.",
    )
    parser.add_argument(
        "--slow-freq",
        type=int,
        default=4,
        help="Chu kỳ cập nhật bộ nhớ chậm f trong thuật toán M3 (mặc định: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Kiểm tra dữ liệu SFT và xuất cấu hình siêu tham số QLoRA mà không cần GPU.",
    )
    args = parser.parse_args()

    return run_qlora_training(
        model_name=args.model_name,
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer_type=args.optimizer,
        slow_freq_f=args.slow_freq,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
