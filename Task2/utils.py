import matplotlib.pyplot as plt

def parse_uploaded_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8")
    return content

def plot_top_ngrams(freq_dist, title, top_n=20):
    items = freq_dist.most_common(top_n)
    words = [' '.join(k) if isinstance(k, tuple) else k for k, _ in items]
    counts = [v for _, v in items]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(words, counts)
    ax.set_title(title)
    ax.set_xticklabels(words, rotation=45, ha="right")
    return fig

def plot_full_freq_distribution(freq_dist):
    sorted_freqs = sorted(freq_dist.values(), reverse=True)
    ranks = range(1, len(sorted_freqs) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ranks, sorted_freqs, marker='.')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Full Frequency Distribution (Log-Log Plot)")
    ax.set_xlabel("Rank of Token")
    ax.set_ylabel("Frequency")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    return fig