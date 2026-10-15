"""tests/test_train.py

REQ: Verify that both packages import and the demo modules run.
WHY: Minimal correctness requirement.

OBS: This repository uses two packages.
     toy_gpt_train_animals (this repository) binds the code to the animals corpus.
     toy_gpt_train (the general repository, a dependency) provides the shared code.
"""


# === This repository: toy_gpt_train_animals ===


def test_local_package_imports() -> None:
    """Test that this repository's package can be imported."""
    import toy_gpt_train_animals

    assert toy_gpt_train_animals is not None


def test_animals_corpus() -> None:
    """Test that the local tokenizer points at the animals corpus file."""
    from toy_gpt_train_animals.a_tokenizer import DEFAULT_CORPUS_PATH

    assert DEFAULT_CORPUS_PATH.name == "001_animals.txt"
    assert DEFAULT_CORPUS_PATH.is_file()


def test_a_tokenizer_runs() -> None:
    """Test that the tokenizer demo runs."""
    from toy_gpt_train_animals import a_tokenizer

    a_tokenizer.main()


def test_b_vocab_runs() -> None:
    """Test that the vocabulary demo runs."""
    from toy_gpt_train_animals import b_vocab

    b_vocab.main()


def test_c_model_runs() -> None:
    """Test that the model demo runs."""
    from toy_gpt_train_animals import c_model

    c_model.main()


def test_d_train_imports() -> None:
    """Test that the training module imports (main() writes artifacts, so no run)."""
    from toy_gpt_train_animals import d_train

    assert d_train.main is not None


def test_e_infer_imports() -> None:
    """Test that the inference module imports."""
    from toy_gpt_train_animals import e_infer

    assert e_infer.main is not None


# === General repository: toy_gpt_train (shared code this repository imports) ===


def test_general_package_imports() -> None:
    """Test that the general package can be imported."""
    import toy_gpt_train

    assert toy_gpt_train is not None


def test_general_tokenizer() -> None:
    """Test that the general package provides SimpleTokenizer."""
    from toy_gpt_train.a_tokenizer import SimpleTokenizer

    assert SimpleTokenizer is not None


def test_general_vocab() -> None:
    """Test that the general package provides Vocabulary."""
    from toy_gpt_train.b_vocab import Vocabulary

    assert Vocabulary is not None


def test_general_model() -> None:
    """Test that the general package provides SimpleNextTokenModel."""
    from toy_gpt_train.c_model import SimpleNextTokenModel

    assert SimpleNextTokenModel is not None


def test_general_training_helpers() -> None:
    """Test that the general package provides the bigram training helpers."""
    from toy_gpt_train.d_train import (
        make_training_pairs,
        row_labeler_bigram,
        train_model,
    )

    assert make_training_pairs is not None
    assert row_labeler_bigram is not None
    assert train_model is not None


def test_general_inference_helpers() -> None:
    """Test that the general package provides the bigram inference helpers."""
    from toy_gpt_train.e_infer import (
        ArtifactVocabulary,
        generate_tokens_bigram,
        load_meta,
        load_model_weights_csv,
        load_vocabulary_csv,
        require_artifacts,
        top_k,
    )

    assert ArtifactVocabulary is not None
    assert generate_tokens_bigram is not None
    assert load_meta is not None
    assert load_model_weights_csv is not None
    assert load_vocabulary_csv is not None
    assert require_artifacts is not None
    assert top_k is not None


def test_general_io_artifacts() -> None:
    """Test that the general package provides the artifact writers."""
    from toy_gpt_train.io_artifacts import write_artifacts, write_training_log

    assert write_artifacts is not None
    assert write_training_log is not None


def test_general_math() -> None:
    """Test that the general package provides argmax."""
    from toy_gpt_train.math_training import argmax

    assert argmax is not None


def test_general_prompts() -> None:
    """Test that the general package provides parse_args."""
    from toy_gpt_train.prompts import parse_args

    assert parse_args is not None
