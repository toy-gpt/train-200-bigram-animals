# Toy-GPT: train-200-bigram-animals

[![Docs](https://img.shields.io/badge/docs-live-blue)](https://toy-gpt.github.io/train-200-bigram-animals/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![CI](https://github.com/toy-gpt/train-200-bigram-animals/actions/workflows/ci-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-200-bigram-animals/actions/workflows/ci-shared.yml)
[![Deploy-Docs](https://github.com/toy-gpt/train-200-bigram-animals/actions/workflows/deploy-docs-shared.yml/badge.svg?branch=main)](https://github.com/toy-gpt/train-200-bigram-animals/actions/workflows/deploy-docs-shared.yml)
[![Check Links](https://github.com/toy-gpt/train-200-bigram-animals/actions/workflows/links.yml/badge.svg)](https://github.com/toy-gpt/train-200-bigram-animals/actions/workflows/links.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/toy-gpt/train-200-bigram-animals/security)

> Demonstrates, at very small scale, how a language model is trained.

This repository is part of a series of toy training repositories plus a companion client repository:

- [**Training repositories**](https://github.com/toy-gpt) produce pretrained artifacts (vocabulary, weights, metadata).
- A [**web app**](https://toy-gpt.github.io/toy-gpt-chat/) loads the artifacts and provides an interactive prompt.

## Contents

- a small, declared text corpus
- a tokenizer and vocabulary builder
- a simple next-token prediction model
- a repeatable training loop
- committed, inspectable artifacts for downstream use

## Scope

This is:

- an intentionally inspectable training pipeline
- a next-token predictor trained on an explicit corpus

This is not:

- a production system
- a full Transformer implementation
- a chat interface
- a claim of semantic understanding

## Outputs

This repository produces and commits pretrained artifacts under `artifacts/`.

Training logs and evidence are written under `outputs/`
(for example, `outputs/train_log.csv`).

## Quick start


Run the full training script:

```shell
uv run python src/toy_gpt_train_animals/d_train.py
```

Run individually:

- a/b/c are demos (can be run alone if desired)
- d_train produces artifacts
- e_infer consumes artifacts

```shell
uv run python src/toy_gpt_train_animals/a_tokenizer.py
uv run python src/toy_gpt_train_animals/b_vocab.py
uv run python src/toy_gpt_train_animals/c_model.py
uv run python src/toy_gpt_train_animals/d_train.py
uv run python src/toy_gpt_train_animals/e_infer.py
```

## Developer

### Clone and Open in VS Code

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/username/train-200-bigram-animals

cd train-200-bigram-animals
code .
```

### Initialize and Update the Python Environment

With the project open in VS Code, open a VS Code terminal.
Paste each command and hit Enter or Return after to run it.

```shell
uvx pup-clean --delete
uv self update
uv python pin 3.14
uv python install
uv lock --upgrade
uv sync
uv audit
```

### Set Up and Run Git Hooks

Set up and run the git hooks to perform
basic checks automatically before
any changes get pushed to GitHub.

In the VS Code terminal,
paste each command and hit Enter or Return after to run it.

```shell
uv run prek install --force
uv run prek update --freeze --cooldown-days 7

git add -A
uv run prek run --all-files
# repeat if changes were made
uv run prek run --all-files
```

### Run Common Chores

Run linters, formatters, type checks, tests, and
build the documentation.

```shell
uv run ruff check . --fix
uv run ruff format .

uv run ty check
uv run python -m pytest
uv run python -m zensical build
```

### Git add-commit-push to GitHub

After making useful changes, save your work to GitHub.

```shell
git add -A
git commit -m "update"
# rerun if changes were made
git add -A
git commit -m "update"

git push -u origin main
```

## Provenance and Purpose

This repository commits pretrained artifacts so the client can run
without retraining.

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
