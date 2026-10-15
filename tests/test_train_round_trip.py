"""tests/test_train_round_trip.py

REQ: Verify training and inference work end to end on the animals corpus.
WHY: Minimal behavioral check for a repository that reuses the general package.

OBS: d_train.main() and e_infer.main() build their paths from the repository
     folder, so this test runs the same steps explicitly and writes everything
     to a temporary folder. It never touches the repository's own artifacts.
"""

from pathlib import Path

from toy_gpt_train.d_train import (
    make_training_pairs,
    row_labeler_bigram,
    train_model,
)
from toy_gpt_train.e_infer import (
    generate_tokens_bigram,
    load_meta,
    load_model_weights_csv,
    load_vocabulary_csv,
    require_artifacts,
)
from toy_gpt_train.io_artifacts import write_artifacts, write_training_log

from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH, SimpleTokenizer
from toy_gpt_train_animals.b_vocab import Vocabulary
from toy_gpt_train_animals.c_model import SimpleNextTokenModel


def test_corpus_is_the_animals_corpus() -> None:
    """Test that the local tokenizer reads this repository's animals corpus."""
    assert DEFAULT_CORPUS_PATH.name == "001_animals.txt"
    assert DEFAULT_CORPUS_PATH.is_file()

    tokens: list[str] = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH).get_tokens()
    assert "tabby" in tokens


def test_train_then_infer_round_trip(tmp_path: Path) -> None:
    """Test training into a temporary folder, then inference from those artifacts."""
    learning_rate: float = 0.1
    epochs: int = 50

    # === Train (the same steps as d_train.main()) ===

    tokens: list[str] = SimpleTokenizer(corpus_path=DEFAULT_CORPUS_PATH).get_tokens()
    vocab: Vocabulary = Vocabulary(tokens)

    token_ids: list[int] = []
    for token in tokens:
        token_id: int | None = vocab.get_token_id(token)
        assert token_id is not None
        token_ids.append(token_id)

    pairs: list[tuple[int, int]] = make_training_pairs(token_ids)
    assert len(pairs) > 0

    model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=vocab.vocab_size())
    history: list[dict[str, float]] = train_model(
        model=model,
        pairs=pairs,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    assert history[-1]["avg_loss"] < history[0]["avg_loss"]

    write_training_log(tmp_path / "outputs" / "train_log.csv", history)
    write_artifacts(
        base_dir=tmp_path,
        corpus_path=DEFAULT_CORPUS_PATH,
        vocab=vocab,
        model=model,
        model_kind="bigram",
        learning_rate=learning_rate,
        epochs=epochs,
        row_labeler=row_labeler_bigram(vocab, vocab.vocab_size()),
    )

    artifacts_dir: Path = tmp_path / "artifacts"
    meta_path: Path = artifacts_dir / "00_meta.json"
    vocab_path: Path = artifacts_dir / "01_vocabulary.csv"
    weights_path: Path = artifacts_dir / "02_model_weights.csv"

    assert (tmp_path / "outputs" / "train_log.csv").is_file()
    assert meta_path.is_file()
    assert vocab_path.is_file()
    assert weights_path.is_file()

    # === Infer (the same steps as e_infer.main()) ===

    require_artifacts(
        meta_path=meta_path,
        vocab_path=vocab_path,
        weights_path=weights_path,
        train_hint="uv run python src/toy_gpt_train_animals/d_train.py",
    )

    meta = load_meta(meta_path)
    assert meta["model_kind"] == "bigram"

    loaded_vocab = load_vocabulary_csv(vocab_path)
    assert loaded_vocab.vocab_size() == vocab.vocab_size()

    v: int = loaded_vocab.vocab_size()
    loaded_model: SimpleNextTokenModel = SimpleNextTokenModel(vocab_size=v)
    loaded_model.weights = load_model_weights_csv(
        weights_path, vocab_size=v, expected_rows=v
    )

    first_id: int = min(loaded_vocab.id_to_token.keys())
    start_token: str = loaded_vocab.id_to_token[first_id]

    generated: list[str] = generate_tokens_bigram(
        model=loaded_model,
        vocab=loaded_vocab,
        start_token=start_token,
        num_tokens=10,
    )
    assert len(generated) > 0
