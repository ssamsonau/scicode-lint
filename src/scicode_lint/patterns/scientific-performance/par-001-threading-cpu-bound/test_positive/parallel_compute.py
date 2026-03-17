import threading


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def parallel_prime_count(numbers, num_threads=4):
    results = {}
    chunk_size = len(numbers) // num_threads

    def worker(start, end):
        count = 0
        for n in numbers[start:end]:
            if is_prime(n):
                count += 1
        results[start] = count

    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(numbers)
        t = threading.Thread(target=worker, args=(start, end))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return sum(results.values())


candidates = list(range(2, 100000))
total_primes = parallel_prime_count(candidates)
