import argparse
from pathlib import Path

from model.config import LabelConfig, Stage1Config
from model.training import train_stage1


def parse_args():
    p = argparse.ArgumentParser(description="Train stage 1 token classifier (ESM)")
    p.add_argument("--train-fasta", type=Path, default=Path("stage1_train.3line.fasta"))
    p.add_argument("--val-fasta", type=Path, default=Path("stage1_test.3line.fasta"))
    p.add_argument("--output-dir", type=Path, default=Path("esm_stage1"))
    p.add_argument("--model-name", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--use-class-weights", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=args.max_len)
    cfg = Stage1Config(
        model_name=args.model_name,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        output_dir=args.output_dir,
        use_class_weights=args.use_class_weights,
    )
    best = train_stage1(cfg, labels)
    print(f"[stage1] best model saved to: {best}")


if __name__ == "__main__":
    main()
