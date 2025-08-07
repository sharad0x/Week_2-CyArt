import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeseries_utils import (
    rolling_mean_np, rolling_mean_numba, rolling_stats_pandas,
    ewma_np, ewma_pandas,
    fft_transform, band_pass_filter
)

def generate_data(n_rows=1_000_000, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.randn(n_rows), name="sensor_reading")

def benchmark_function(func, *args):
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return end - start, result

def run_benchmarks():
    sizes = [10_000, 100_000, 1_000_000]
    results = []

    for size in sizes:
        print(f"=== Benchmarking size: {size} ===")
        data = generate_data(size)
        window = 50
        alpha = 0.1

        t_np, _ = benchmark_function(rolling_mean_np, data.values, window)
        t_numba, _ = benchmark_function(rolling_mean_numba, data.values, window)
        t_pd, _ = benchmark_function(lambda s, w: s.rolling(w).mean(), data, window)

        t_ewma_np, _ = benchmark_function(ewma_np, data.values, alpha)
        t_ewma_pd, _ = benchmark_function(ewma_pandas, data, alpha)

        results.append({
            "size": size,
            "rolling_np": t_np,
            "rolling_numba": t_numba,
            "rolling_pandas": t_pd,
            "ewma_np": t_ewma_np,
            "ewma_pandas": t_ewma_pd
        })

    return pd.DataFrame(results)

def plot_results(df):
    df.set_index("size", inplace=True)
    df.plot(kind="bar", figsize=(12, 6), logy=True, rot=0)
    plt.ylabel("Execution Time (seconds, log scale)")
    plt.title("Performance Benchmark: NumPy vs Numba vs pandas")
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    df = run_benchmarks()
    df.to_csv("benchmark_results.csv", index=False)
    plot_results(df)
