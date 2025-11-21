import argparse
from pathlib import Path

from model.config import LabelConfig, NestedCVConfig
from model.nested_cv import run_nested_cv


def parse_args():
    p = argparse.ArgumentParser(description="Run nested CV for stage2 hyperparameters")
    p.add_argument("--base-model", type=Path, default=Path("esm_stage1/best_model"))
    p.add_argument("--train-fasta", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    p.add_argument("--val-fasta", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    p.add_argument("--output-dir", type=Path, default=Path("esm_stage2_cv_outputs"))
    p.add_argument("--max-len", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=args.max_len)
    cfg = NestedCVConfig(
        base_model_path=args.base_model,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        output_dir=args.output_dir,
    )
    best_hp, mean_f1 = run_nested_cv(cfg, labels)
    print(f"[nested-cv] best HPs: {best_hp}")
    print(f"[nested-cv] mean outer f1_M: {mean_f1:.4f}")


if __name__ == "__main__":
    main()
