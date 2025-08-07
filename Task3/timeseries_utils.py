import numpy as np
import pandas as pd
from numba import njit

def rolling_mean_np(arr, window):
    shape = (arr.shape[0] - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return windows.mean(axis=1)

def rolling_var_np(arr, window):
    shape = (arr.shape[0] - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return windows.var(axis=1)

def rolling_stats_pandas(series, window):
    return series.rolling(window).mean(), series.rolling(window).var()

@njit
def rolling_mean_numba(arr, window):
    n = len(arr)
    out = np.empty(n - window + 1)
    for i in range(n - window + 1):
        out[i] = arr[i:i + window].mean()
    return out

@njit
def rolling_var_numba(arr, window):
    n = len(arr)
    out = np.empty(n - window + 1)
    for i in range(n - window + 1):
        out[i] = arr[i:i + window].var()
    return out

def ewma_np(arr, alpha):
    ewma = np.empty_like(arr)
    ewma[0] = arr[0]
    for t in range(1, len(arr)):
        ewma[t] = alpha * arr[t] + (1 - alpha) * ewma[t-1]
    return ewma

def ewma_pandas(series, alpha):
    return series.ewm(alpha=alpha).mean()

def ewm_cov_np(x, y, alpha):
    ewma_x = ewma_np(x, alpha)
    ewma_y = ewma_np(y, alpha)
    cov = np.empty_like(x)
    for t in range(len(x)):
        cov[t] = alpha * (x[t] - ewma_x[t]) * (y[t] - ewma_y[t])
    return cov

def fft_transform(arr):
    """
    Compute the Fast Fourier Transform of the input 1D array.
    Returns complex frequency coefficients.
    """
    return np.fft.fft(arr)

def band_pass_filter(arr, low, high, fs):
    """
    Apply band-pass filter using FFT.
    Arguments:
        arr: input signal (1D numpy array)
        low: low cutoff frequency (Hz)
        high: high cutoff frequency (Hz)
        fs: sampling frequency (Hz)
    Returns:
        filtered signal (real part of inverse FFT)
    """
    n = len(arr)
    fft_data = np.fft.fft(arr)
    freqs = np.fft.fftfreq(n, d=1/fs)
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    fft_data[~mask] = 0
    filtered_signal = np.fft.ifft(fft_data)
    return filtered_signal.real

def auto_rolling_mean(arr_or_series, window):
    if isinstance(arr_or_series, pd.Series):
        size = len(arr_or_series)
    else:
        size = arr_or_series.size

    if size > 1_000_000:
        return rolling_mean_numba(arr_or_series.values if isinstance(arr_or_series, pd.Series) else arr_or_series, window)
    elif size > 100_000:
        return rolling_mean_np(arr_or_series.values if isinstance(arr_or_series, pd.Series) else arr_or_series, window)
    else:
        return rolling_stats_pandas(arr_or_series, window)[0]
