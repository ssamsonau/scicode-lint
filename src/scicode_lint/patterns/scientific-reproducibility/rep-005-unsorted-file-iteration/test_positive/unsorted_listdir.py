"""Text corpus loading with unsorted file iteration."""

import os
from pathlib import Path


def load_all_data(corpus_dir):
    """Load text files and build token list - order affects concatenation."""
    all_tokens = []
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            with open(filepath) as f:
                tokens = f.read().split()
                all_tokens.extend(tokens)
    return all_tokens


def build_vocab(corpus_dir: Path) -> dict[str, int]:
    """Build vocabulary from corpus - word IDs depend on file order."""
    vocab = {}
    idx = 0
    for doc_path in corpus_dir.iterdir():
        if doc_path.suffix == ".txt":
            with open(doc_path) as f:
                for word in f.read().split():
                    if word not in vocab:
                        vocab[word] = idx
                        idx += 1
    return vocab
