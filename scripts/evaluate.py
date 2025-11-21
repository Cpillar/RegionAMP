import argparse
from pathlib import Path

from model.config import EvalConfig, LabelConfig
from model.eval import evaluate_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate stage2 model on test set")
    p.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    p.add_argument("--base-config", type=Path, default=None)
    p.add_argument("--test-fasta", type=Path, default=Path("test_dataset_stage2.3line.fasta"))
    p.add_argument("--output-dir", type=Path, default=Path("esm_stage2_crf/evaluation_results"))
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=args.max_len)
    cfg = EvalConfig(
        model_dir=args.model_dir,
        base_config_dir=args.base_config,
        test_fasta=args.test_fasta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    metrics = evaluate_model(cfg, labels)
    print(f"[eval] metrics: {metrics}")


if __name__ == "__main__":
    main()
