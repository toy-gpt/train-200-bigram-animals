"""e_infer.py - Inference module (artifact-driven).

Runs inference using previously saved training artifacts.

Responsibilities:
- Load inspectable training artifacts from artifacts/
  - 00_meta.json
  - 01_vocabulary.csv
  - 02_model_weights.csv
- Reconstruct a vocabulary-like interface and model weights
- Generate tokens using greedy decoding (argmax)
- Print top-k next-token probabilities for inspection

Notes:
- This module does NOT retrain by default.
- If artifacts are missing, run d_train.py first.
"""

import argparse
import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header

from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.e_infer import (
    ArtifactVocabulary,
    generate_tokens_bigram,
    load_meta,
    load_model_weights_csv,
    load_vocabulary_csv,
    require_artifacts,
    top_k,
)
from toy_gpt_train.prompts import parse_args

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

LOG: logging.Logger = get_logger("INFER", level="INFO")


def main() -> None:
    """Run inference using saved training artifacts."""
    log_header(LOG, "Inference Demo: Load Artifacts and Generate Text")

    base_dir: Final[Path] = Path(__file__).resolve().parents[2]
    artifacts_dir: Final[Path] = base_dir / "artifacts"
    meta_path: Final[Path] = artifacts_dir / "00_meta.json"
    vocab_path: Final[Path] = artifacts_dir / "01_vocabulary.csv"
    weights_path: Final[Path] = artifacts_dir / "02_model_weights.csv"
    require_artifacts(
        meta_path=meta_path,
        vocab_path=vocab_path,
        weights_path=weights_path,
        train_hint="uv run python src/toy_gpt_train_animals/d_train.py",
    )

    meta: JsonObject = load_meta(meta_path)
    vocab: ArtifactVocabulary = load_vocabulary_csv(vocab_path)

    v = vocab.vocab_size()
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=v)
    model.weights = load_model_weights_csv(weights_path, vocab_size=v, expected_rows=v)

    args: argparse.Namespace = parse_args([])

    # Choose a start token.
    start_token = args.start_token
    if not start_token:
        # Deterministic fallback: smallest token_id present
        first_id = min(vocab.id_to_token.keys())
        start_token = vocab.id_to_token[first_id]

    LOG.info(
        f"Loaded repo_name={meta.get('repo_name')} model_kind={meta.get('model_kind')}"
    )
    LOG.info(f"Vocab size: {vocab.vocab_size()}")
    LOG.info(f"Start token: {start_token!r}")

    start_id = vocab.get_token_id(start_token)
    if start_id is not None:
        probs: list[float] = model.forward(start_id)
        LOG.info(f"Top next-token predictions after {start_token!r}:")
        for tok_id, prob in top_k(probs, k=max(1, args.topk)):
            tok = vocab.get_id_token(tok_id)
            LOG.info(f"  {tok!r} (ID {tok_id}): {prob:.4f}")

    generated = generate_tokens_bigram(
        model=model,
        vocab=vocab,
        start_token=start_token,
        num_tokens=max(0, args.num_tokens),
    )

    LOG.info("Generated sequence:")
    LOG.info("  " + " ".join(generated))


if __name__ == "__main__":
    main()
