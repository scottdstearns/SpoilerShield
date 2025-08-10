import argparse
import os
from typing import List

import joblib
import numpy as np


def load_baseline_pipeline(baseline_path: str):
    """Load baseline model pipeline from a .pkl file.

    The saved artifact may be either a dict that contains a 'pipeline' key
    or the pipeline object directly. This helper normalizes to returning the
    pipeline object.
    """
    obj = joblib.load(baseline_path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    return obj


def predict_with_baseline(pipeline, texts: List[str]):
    # Prefer predict_proba if available for calibrated probabilities
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)
        # Assume positive class is at index 1
        pos_scores = proba[:, 1]
    else:
        # Fall back to decision_function and squash to (0,1)
        decision = pipeline.decision_function(texts)
        pos_scores = 1.0 / (1.0 + np.exp(-decision))

    preds = (pos_scores >= 0.5).astype(int)
    return preds, pos_scores


def predict_with_transformer(model_dir: str, texts: List[str]):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = logits.softmax(dim=-1).detach().cpu().numpy()

    preds = (probs[:, 1] >= 0.5).astype(int)
    return preds, probs[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Quick inference for SpoilerShield")
    parser.add_argument(
        "--model",
        choices=["baseline", "transformer", "both"],
        default="baseline",
        help="Which model to run",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "I won't spoil the ending, but it's a twist!",
            "This review reveals exactly what happens in the final scene.",
        ],
        help="Text(s) to classify",
    )
    parser.add_argument(
        "--baseline-path",
        default=os.path.join("outputs", "baseline_model.pkl"),
        help="Path to baseline .pkl artifact",
    )
    parser.add_argument(
        "--transformer-dir",
        default=os.path.join("outputs", "roberta-base_model"),
        help="Directory containing fine-tuned transformer artifacts",
    )
    args = parser.parse_args()

    texts = args.text

    if args.model in ("baseline", "both"):
        pipeline = load_baseline_pipeline(args.baseline_path)
        preds, scores = predict_with_baseline(pipeline, texts)
        for t, y, s in zip(texts, preds, scores):
            print(f"[Baseline] label={int(y)} prob_spoiler={float(s):.4f} text={t}")

    if args.model in ("transformer", "both"):
        if not os.path.isdir(args.transformer_dir):
            raise SystemExit(
                f"Transformer dir not found: {args.transformer_dir}. See README for download instructions."
            )
        preds, scores = predict_with_transformer(args.transformer_dir, texts)
        for t, y, s in zip(texts, preds, scores):
            print(f"[Transformer] label={int(y)} prob_spoiler={float(s):.4f} text={t}")


if __name__ == "__main__":
    main()


