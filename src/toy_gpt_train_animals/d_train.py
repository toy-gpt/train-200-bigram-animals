"""d_train.py - Training loop module.

Trains the SimpleNextTokenModel on a small token corpus
using a bigram context (current token → next token).

Responsibilities:
- Create (current_token -> next_token) training pairs from the corpus
- Run a basic gradient-descent training loop
- Track loss and accuracy per epoch
- Write a CSV log of training progress
- Write inspectable training artifacts (vocabulary, weights, embeddings, meta)

Concepts:
- epoch: one complete pass through all training pairs
- softmax: converts raw scores into probabilities (so predictions sum to 1)
- cross-entropy loss: measures how well predicted probabilities match the correct token
- gradient descent: iterative weight updates to minimize loss
  - think descending to find the bottom of a valley in a landscape
  - where the valley floor corresponds to lower prediction error

Notes:
- This is intentionally simple: no deep learning framework, no Transformer.
- The model is a softmax regression classifier over bigram contexts.
- Training updates the weight row corresponding to the observed current token.
- token_embeddings.csv is a visualization-friendly projection for levels 100-400;
  in later repos (500+), embeddings become a first-class learned table.
"""

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from toy_gpt_train.c_model import SimpleNextTokenModel
from toy_gpt_train.d_train import (
    make_training_pairs,
    row_labeler_bigram,
    train_model,
)
from toy_gpt_train.io_artifacts import (
    write_artifacts,
    write_training_log,
)
from toy_gpt_train.math_training import argmax

from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
from toy_gpt_train_animals.b_vocab import Vocabulary

type BigramPair = tuple[int, int]

LOG: logging.Logger = get_logger("TRAIN", level="INFO")


def main() -> None:
    """Run a simple training demo end-to-end."""
    log_header(LOG, "Training Demo: Next-Token Softmax Regression")

    base_dir: Final[Path] = Path(__file__).resolve().parents[2]
    outputs_dir: Final[Path] = base_dir / "outputs"
    train_log_path: Final[Path] = outputs_dir / "train_log.csv"

    # Step 1: Load and tokenize the corpus.
    tokenizer: SimpleTokenizer = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH)
    tokens: list[str] = tokenizer.get_tokens()

    if not tokens:
        LOG.error("No tokens found. Check corpus file.")
        return

    # Step 2: Build vocabulary (maps tokens <-> integer IDs).
    vocab: Vocabulary = Vocabulary(tokens)

    # Step 3: Convert token strings to integer IDs for training.
    token_ids: list[int] = []
    for tok in tokens:
        tok_id: int | None = vocab.get_token_id(tok)
        if tok_id is None:
            LOG.error(f"Token not found in vocabulary: {tok!r}")
            return
        token_ids.append(tok_id)

    # Step 4: Create training pairs (current -> next).
    pairs: list[BigramPair] = make_training_pairs(token_ids)
    LOG.info(f"Created {len(pairs)} training pairs.")

    # Step 5: Initialize model with random weights.
    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())

    # Step 6: Train the model.
    learning_rate: float = 0.1
    epochs: int = 50

    history: list[dict[str, float]] = train_model(
        model=model,
        pairs=pairs,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    # Step 7: Save training metrics for analysis.
    write_training_log(train_log_path, history)

    # Step 7b: Write inspectable artifacts for downstream use.
    write_artifacts(
        base_dir=base_dir,
        corpus_path=DEFAULT_CORPUS_PATH,
        vocab=vocab,
        model=model,
        model_kind="bigram",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_bigram(vocab, vocab.vocab_size()),
    )

    # Step 8: Qualitative check - what does the model predict after first token?
    current_token: str = tokens[0]
    current_id: int | None = vocab.get_token_id(current_token)
    if current_id is not None:
        probs: list[float] = model.forward(current_id)
        best_next_id: int = argmax(probs)
        best_next_tok: str | None = vocab.get_id_token(best_next_id)
        LOG.info(
            f"After training, most likely next token after {current_token!r} "
            f"is {best_next_tok!r} (ID: {best_next_id})."
        )


if __name__ == "__main__":
    main()
