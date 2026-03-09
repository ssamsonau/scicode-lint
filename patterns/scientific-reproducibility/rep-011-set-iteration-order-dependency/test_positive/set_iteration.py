def process(item):
    return item * 2


def process_unique_items(items):
    unique_items = set(items)
    results = []
    for item in unique_items:
        results.append(process(item))
    return results


def get_unique_features(data):
    features = set()
    for row in data:
        features.update(row.keys())
    return list(features)


def build_vocabulary(texts):
    vocab = set()
    for text in texts:
        vocab.update(text.split())

    word_to_idx = {}
    for i, word in enumerate(vocab):
        word_to_idx[word] = i
    return word_to_idx
