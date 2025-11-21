import argparse
from pathlib import Path

from model.config import EvalConfig, LabelConfig
from model.eval import evaluate_model_iou_threshold


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate IoU threshold for target label (default M)")
    p.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    p.add_argument("--test-fasta", type=Path, default=Path("test_dataset_stage2.3line.fasta"))
    p.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf/evaluation_results_iou"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target-label", type=str, default="M")
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=args.max_len)
    cfg = EvalConfig(
        model_dir=args.model_dir,
        base_config_dir=None,
        test_fasta=args.test_fasta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        iou_threshold=args.threshold,
    )
    res = evaluate_model_iou_threshold(cfg, labels, target_label=args.target_label)
    print(f"[eval-iou] summary: {res['summary']}")


if __name__ == "__main__":
    main()
