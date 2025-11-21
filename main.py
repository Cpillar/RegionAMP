"""
Top-level orchestrator for AMP region workflows.

Provides a single entrypoint to run individual steps or a simple end-to-end
pipeline (stage1 -> stage2 -> eval). Each step delegates to the underlying
library functions in the `model` package.
"""

import argparse
from pathlib import Path
from typing import Optional

from model.config import (
    EvalConfig,
    LabelConfig,
    NestedCVConfig,
    PredictionConfig,
    Stage1Config,
    Stage2Config,
)
from model.eval import evaluate_model, evaluate_model_iou_threshold
from model.nested_cv import run_nested_cv
from model.predict import predict_fasta
from model.training import train_stage1, train_stage2


def run_stage1(args, labels: LabelConfig) -> Path:
    cfg = Stage1Config(
        model_name=args.model_name,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        output_dir=args.output_dir,
    )
    return train_stage1(cfg, labels)


def run_stage2(args, labels: LabelConfig) -> Path:
    cfg = Stage2Config(
        base_model_path=args.base_model,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        output_dir=args.output_dir,
    )
    return train_stage2(cfg, labels)


def run_nested(args, labels: LabelConfig):
    cfg = NestedCVConfig(
        base_model_path=args.base_model,
        train_fasta=args.train_fasta,
        val_fasta=args.val_fasta,
        output_dir=args.output_dir,
    )
    return run_nested_cv(cfg, labels)


def run_eval(args, labels: LabelConfig):
    cfg = EvalConfig(
        model_dir=args.model_dir,
        base_config_dir=args.base_config,
        test_fasta=args.test_fasta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    return evaluate_model(cfg, labels)


def run_iou(args, labels: LabelConfig):
    cfg = EvalConfig(
        model_dir=args.model_dir,
        base_config_dir=None,
        test_fasta=args.test_fasta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        iou_threshold=args.threshold,
    )
    return evaluate_model_iou_threshold(cfg, labels, target_label=args.target_label)


def run_predict(args, labels: LabelConfig):
    cfg = PredictionConfig(
        model_dir=args.model_dir,
        fasta_path=args.fasta,
        output_path=args.output,
        max_len=args.max_len,
    )
    return predict_fasta(cfg, labels)


def parse_args():
    p = argparse.ArgumentParser(description="Main orchestrator for AMP region workflow")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared label arg
    def add_common(pp):
        pp.add_argument("--max-len", type=int, default=200)

    # Stage1
    s1 = sub.add_parser("stage1", help="Train stage1 token classifier")
    s1.add_argument("--train-fasta", type=Path, default=Path("stage1_train.3line.fasta"))
    s1.add_argument("--val-fasta", type=Path, default=Path("stage1_test.3line.fasta"))
    s1.add_argument("--output-dir", type=Path, default=Path("esm_stage1"))
    s1.add_argument("--model-name", type=str, default="facebook/esm2_t12_35M_UR50D")
    add_common(s1)

    # Stage2
    s2 = sub.add_parser("stage2", help="Train stage2 ESM+CRF")
    s2.add_argument("--base-model", type=Path, default=Path("esm_stage1/best_model"))
    s2.add_argument("--train-fasta", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    s2.add_argument("--val-fasta", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    s2.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf"))
    add_common(s2)

    # Nested CV
    nc = sub.add_parser("nested-cv", help="Run nested CV for stage2 HPs")
    nc.add_argument("--base-model", type=Path, default=Path("esm_stage1/best_model"))
    nc.add_argument("--train-fasta", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    nc.add_argument("--val-fasta", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    nc.add_argument("--output-dir", type=Path, default=Path("esm_stage2_cv_outputs"))
    add_common(nc)

    # Eval
    ev = sub.add_parser("eval", help="Evaluate model on test set")
    ev.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    ev.add_argument("--base-config", type=Path, default=None)
    ev.add_argument("--test-fasta", type=Path, default=Path("test_dataset_stage2.3line.fasta"))
    ev.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf/evaluation_results"))
    ev.add_argument("--batch-size", type=int, default=16)
    add_common(ev)

    # IoU Eval
    iou = sub.add_parser("eval-iou", help="Evaluate per-sequence IoU threshold")
    iou.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    iou.add_argument("--test-fasta", type=Path, default=Path("test_dataset_stage2.3line.fasta"))
    iou.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf/evaluation_results_iou"))
    iou.add_argument("--batch-size", type=int, default=16)
    iou.add_argument("--threshold", type=float, default=0.5)
    iou.add_argument("--target-label", type=str, default="M")
    add_common(iou)

    # Predict
    pr = sub.add_parser("predict", help="Predict labels for FASTA")
    pr.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    pr.add_argument("--fasta", type=Path, required=True)
    pr.add_argument("--output", type=Path, default=Path("predictions.tsv"))
    add_common(pr)

    # Pipeline: stage1 -> stage2 (optional eval)
    pipe = sub.add_parser("pipeline", help="Run stage1 then stage2; optional eval")
    pipe.add_argument("--stage1-train", type=Path, default=Path("stage1_train.3line.fasta"))
    pipe.add_argument("--stage1-val", type=Path, default=Path("stage1_test.3line.fasta"))
    pipe.add_argument("--stage1-out", type=Path, default=Path("esm_stage1"))
    pipe.add_argument("--stage2-train", type=Path, default=Path("train_dataset_stage2.3line.fasta"))
    pipe.add_argument("--stage2-val", type=Path, default=Path("validation_dataset_stage2.3line.fasta"))
    pipe.add_argument("--stage2-out", type=Path, default=Path("esm_stage2_crf"))
    pipe.add_argument("--eval-test", type=Path, default=None, help="Optional test fasta to eval after stage2")
    pipe.add_argument("--eval-output", type=Path, default=Path("esm_stage2_crf/evaluation_results"))
    add_common(pipe)

    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=getattr(args, "max_len", 200))

    if args.cmd == "stage1":
        best = run_stage1(args, labels)
        print(f"[main][stage1] best model saved to {best}")
    elif args.cmd == "stage2":
        best = run_stage2(args, labels)
        print(f"[main][stage2] best model saved to {best}")
    elif args.cmd == "nested-cv":
        best_hp, mean_f1 = run_nested(args, labels)
        print(f"[main][nested-cv] best HP: {best_hp}, mean outer f1_M={mean_f1:.4f}")
    elif args.cmd == "eval":
        metrics = run_eval(args, labels)
        print(f"[main][eval] metrics: {metrics}")
    elif args.cmd == "eval-iou":
        res = run_iou(args, labels)
        print(f"[main][eval-iou] summary: {res['summary']}")
    elif args.cmd == "predict":
        out = run_predict(args, labels)
        print(f"[main][predict] predictions saved to: {out}")
    elif args.cmd == "pipeline":
        # Stage1
        s1_args = argparse.Namespace(
            model_name="facebook/esm2_t12_35M_UR50D",
            train_fasta=args.stage1_train,
            val_fasta=args.stage1_val,
            output_dir=args.stage1_out,
            max_len=args.max_len,
        )
        best_stage1 = run_stage1(s1_args, labels)
        print(f"[main][pipeline] stage1 best: {best_stage1}")

        # Stage2 (uses stage1 best)
        s2_args = argparse.Namespace(
            base_model=best_stage1,
            train_fasta=args.stage2_train,
            val_fasta=args.stage2_val,
            output_dir=args.stage2_out,
            max_len=args.max_len,
        )
        best_stage2 = run_stage2(s2_args, labels)
        print(f"[main][pipeline] stage2 best: {best_stage2}")

        # Optional eval
        if args.eval_test:
            ev_args = argparse.Namespace(
                model_dir=best_stage2,
                base_config=None,
                test_fasta=args.eval_test,
                output_dir=args.eval_output,
                batch_size=16,
                max_len=args.max_len,
            )
            metrics = run_eval(ev_args, labels)
            print(f"[main][pipeline] eval metrics: {metrics}")
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
