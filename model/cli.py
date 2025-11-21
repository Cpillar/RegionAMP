"""Command-line interface for AMP region workflows."""

import argparse
from pathlib import Path

from .config import (
    EvalConfig,
    LabelConfig,
    NestedCVConfig,
    PredictionConfig,
    Stage1Config,
    Stage2Config,
)
from .eval import evaluate_model, evaluate_model_iou_threshold
from .nested_cv import run_nested_cv
from .predict import predict_fasta
from .training import train_stage1, train_stage2


def main():
    parser = argparse.ArgumentParser(description="AMP region CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train-stage1", help="Train stage 1 token classifier")
    p1.add_argument("--train-fasta", type=Path, default=Path("stage1_train.3line.fasta"))
    p1.add_argument("--val-fasta", type=Path, default=Path("stage1_test.3line.fasta"))
    p1.add_argument("--output-dir", type=Path, default=Path("esm_stage1"))
    p1.add_argument("--model-name", type=str, default="facebook/esm2_t12_35M_UR50D")
    p1.add_argument("--max-len", type=int, default=200)

    p2 = sub.add_parser("train-stage2", help="Train stage 2 ESM+CRF")
    p2.add_argument("--base-model", type=Path, default=Path("esm_stage1/best_model"))
    p2.add_argument("--train-fasta", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    p2.add_argument("--val-fasta", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    p2.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf"))
    p2.add_argument("--max-len", type=int, default=200)

    p3 = sub.add_parser("nested-cv", help="Run nested CV for stage 2")
    p3.add_argument("--base-model", type=Path, default=Path("esm_stage1/best_model"))
    p3.add_argument("--train-fasta", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    p3.add_argument("--val-fasta", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    p3.add_argument("--output-dir", type=Path, default=Path("esm_stage2_cv_outputs"))
    p3.add_argument("--max-len", type=int, default=200)

    p4 = sub.add_parser("eval", help="Evaluate model on test set")
    p4.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    p4.add_argument("--base-config", type=Path, default=None)
    p4.add_argument("--test-fasta", type=Path, default=Path("test_dataset_stage2.3line.fasta"))
    p4.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf/evaluation_results"))
    p4.add_argument("--max-len", type=int, default=200)
    p4.add_argument("--batch-size", type=int, default=16)

    p5 = sub.add_parser("eval-iou", help="Evaluate per-sequence IoU")
    p5.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    p5.add_argument("--test-fasta", type=Path, default=Path("test_dataset_stage2.3line.fasta"))
    p5.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf/evaluation_results_iou"))
    p5.add_argument("--threshold", type=float, default=0.5)
    p5.add_argument("--target-label", type=str, default="M")
    p5.add_argument("--max-len", type=int, default=200)
    p5.add_argument("--batch-size", type=int, default=16)

    p6 = sub.add_parser("predict", help="Predict labels for FASTA")
    p6.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    p6.add_argument("--fasta", type=Path, required=True)
    p6.add_argument("--output", type=Path, default=Path("predictions.tsv"))
    p6.add_argument("--max-len", type=int, default=200)

    args = parser.parse_args()
    labels = LabelConfig(max_len=getattr(args, "max_len", 200))

    if args.cmd == "train-stage1":
        cfg = Stage1Config(
            model_name=args.model_name,
            train_fasta=args.train_fasta,
            val_fasta=args.val_fasta,
            output_dir=args.output_dir,
        )
        best = train_stage1(cfg, labels)
        print(f"[cli][stage1] best model saved to {best}")

    elif args.cmd == "train-stage2":
        cfg = Stage2Config(
            base_model_path=args.base_model,
            train_fasta=args.train_fasta,
            val_fasta=args.val_fasta,
            output_dir=args.output_dir,
        )
        best = train_stage2(cfg, labels)
        print(f"[cli][stage2] best model saved to {best}")

    elif args.cmd == "nested-cv":
        cfg = NestedCVConfig(
            base_model_path=args.base_model,
            train_fasta=args.train_fasta,
            val_fasta=args.val_fasta,
            output_dir=args.output_dir,
        )
        best, mean_f1 = run_nested_cv(cfg, labels)
        print(f"[cli][nested-cv] best HP: {best}, mean outer f1_M={mean_f1:.4f}")

    elif args.cmd == "eval":
        cfg = EvalConfig(
            model_dir=args.model_dir,
            base_config_dir=args.base_config,
            test_fasta=args.test_fasta,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        metrics = evaluate_model(cfg, labels)
        print(f"[cli][eval] metrics: {metrics}")

    elif args.cmd == "eval-iou":
        cfg = EvalConfig(
            model_dir=args.model_dir,
            base_config_dir=None,
            test_fasta=args.test_fasta,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            iou_threshold=args.threshold,
        )
        res = evaluate_model_iou_threshold(cfg, labels, target_label=args.target_label)
        print(f"[cli][eval-iou] summary: {res['summary']}")

    elif args.cmd == "predict":
        cfg = PredictionConfig(
            model_dir=args.model_dir,
            fasta_path=args.fasta,
            output_path=args.output,
            max_len=args.max_len,
        )
        out = predict_fasta(cfg, labels)
        print(f"[cli][predict] predictions saved to: {out}")


if __name__ == "__main__":
    main()
