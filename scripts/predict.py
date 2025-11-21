import argparse
from pathlib import Path

from model.config import LabelConfig, PredictionConfig
from model.predict import predict_fasta


def parse_args():
    p = argparse.ArgumentParser(description="Predict labels for sequences in a FASTA file")
    p.add_argument("--model-dir", type=Path, default=Path("esm_stage2_crf/best_model"))
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("predictions.tsv"))
    p.add_argument("--max-len", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    labels = LabelConfig(max_len=args.max_len)
    cfg = PredictionConfig(
        model_dir=args.model_dir,
        fasta_path=args.fasta,
        output_path=args.output,
        max_len=args.max_len,
    )
    out = predict_fasta(cfg, labels)
    print(f"[predict] predictions saved to: {out}")


if __name__ == "__main__":
    main()
