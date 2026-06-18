from concurrent.futures import ThreadPoolExecutor, as_completed


def run_parallel(fn, items, max_workers=5):
    """Run fn(item) for each item in parallel. Returns results in input order."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
    return results
