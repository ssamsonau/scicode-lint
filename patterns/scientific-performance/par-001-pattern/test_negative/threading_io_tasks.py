import threading
import time


def fetch_data_from_api(url, results, idx):
    time.sleep(0.1)
    results[idx] = f"Data from {url}"


def parallel_data_fetch(urls):
    threads = []
    results = {}

    for i, url in enumerate(urls):
        t = threading.Thread(target=fetch_data_from_api, args=(url, results, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return [results[i] for i in range(len(urls))]


urls = [f"http://api.example.com/data/{i}" for i in range(10)]
data = parallel_data_fetch(urls)
