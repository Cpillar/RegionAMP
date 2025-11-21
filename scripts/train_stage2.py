import argparse
from pathlib import Path

from model.config import LabelConfig, Stage2Config
from model.training import train_stage2


def parse_args():
    p = argparse.ArgumentParser(description="Train stage 2 ESM+CRF model")
    p.add_argument("--base-model", type=Path, default=Path("esm_stage1/best_model"))
    p.add_argument("--train-fasta", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    p.add_argument("--val-fasta", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    p.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf"))
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--learning-rate-esm", type=float, default=1e-5)
    p.add_argument("--crf-head-lr-multiplier", type=float, default=10.0)
    p.add_argument("--num-train-epochs", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=args.max_len)
    cfg = Stage2Config(
        base_model_path=args.base_model,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        output_dir=args.output_dir,
        learning_rate_esm=args.learning_rate_esm,
        crf_head_lr_multiplier=args.crf_head_lr_multiplier,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
    )
    best = train_stage2(cfg, labels)
    print(f"[stage2] best model saved to: {best}")


if __name__ == "__main__":
    main()
