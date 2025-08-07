# ⚡ High-Performance Time Series Transformation with NumPy & pandas

This project performs efficient, vectorized time-series transformations (rolling stats, EWMA, FFT analysis) on large multivariate datasets using NumPy, pandas, Numba, and stride tricks. It includes a benchmark comparison to find the fastest method automatically.

---

## 📁 Project Structure

```
.
├── timeseries_utils.py       # Core transformation functions
├── benchmark.py              # Runtime and memory benchmark runner
├── benchmark_results.csv     # Auto-generated after running benchmark.py
├── benchmark_results.png     # Visualization of runtimes
├── report.md                 # Analysis of performance trade-offs
└── README.md                 # Setup and usage guide
```

---

## ⚙️ Setup Instructions

### 1. Clone or Download the Project

```bash
git clone https://github.com/your_username/high-perf-timeseries.git
cd high-perf-timeseries
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Benchmark

```bash
python benchmark.py
```

This will:
- Generate synthetic time series data (10k, 100k, 1M samples)
- Benchmark all implementations (NumPy, Numba, pandas)
- Export results to `benchmark_results.csv`
- Show and save the performance plot as `benchmark_results.png`

---

## 📊 Features Supported

| Feature                         | Implemented via                      |
|----------------------------------|---------------------------------------|
| Rolling Mean / Variance         | NumPy (stride), pandas, Numba        |
| EWMA / Covariance               | Manual exponential smoothing & pandas |
| FFT + Band-Pass Filtering       | NumPy `fft` tools                     |
| Auto Selection (best performer) | Based on dataset size                |

---

## 📌 Key Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `numba`

---

## 🧪 Example Usage

```python
from timeseries_utils import auto_rolling_mean, ewma_np

data = np.random.randn(1_000_000)
mean_out = auto_rolling_mean(data, window=100)

ewma_out = ewma_np(data, alpha=0.05)
```

---

## 📄 Report

For detailed performance trade-offs, memory use, and design recommendations, see [report.md](./report.md).
