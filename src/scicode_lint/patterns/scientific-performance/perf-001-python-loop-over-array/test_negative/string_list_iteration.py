import os


def count_vowels(text):
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count


def process_filenames(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            results.append(filename)
    return results


def parse_csv_row(row_string):
    values = []
    for cell in row_string.split(","):
        values.append(cell.strip())
    return values


def filter_valid_items(items):
    valid = []
    for item in items:
        if item.get("status") == "active":
            valid.append(item)
    return valid


def build_index(documents):
    index = {}
    for doc_id, doc in enumerate(documents):
        for word in doc.split():
            if word not in index:
                index[word] = []
            index[word].append(doc_id)
    return index
