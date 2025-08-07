
# 🧠 NLTK-Powered Text Analytics Web App

This project is a **Streamlit-based NLP analytics dashboard** that allows users to upload plain-text files and explore linguistic insights using **NLTK**, **pandas**, and **matplotlib**.

---

## 🚀 Features

### 1. **Main Page**
- Cleans uploaded text (lowercase, punctuation removal, stop word removal, lemmatization).
- Stores results in a `pandas.DataFrame`.
- Displays processed DataFrame and summary.

### 2. **Data Explorer**
- Token frequency distribution with full-length bar plot.
- Displays collocations (common bigram patterns).
- Sentiment analysis of tokens using NLTK VADER.

### 3. **Analysis Dashboard**
- N-gram visualizations: Unigrams, Bigrams and Trigrams.
- Sentiment trend line plot across text sections.

---

## 📁 Project Structure

```
nlp-text-analyzer/
├── streamlit_app.py       # Streamlit frontend
├── nlp_pipeline.py        # Text processing backend (cleaning, analysis)
├── sample_review.txt      # Sample test file (user can upload custom .txt)
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/nlp-text-analyzer.git
cd nlp-text-analyzer
pip install -r requirements.txt
```

Make sure to download necessary NLTK resources before running:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

---

## ▶️ How to Run

Launch the app locally using Streamlit:

```bash
streamlit run streamlit_app.py
```

Then open the URL in your browser (usually http://localhost:8501).

---

## 🧠 Credits

Built using:
- [NLTK](https://www.nltk.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [Streamlit](https://streamlit.io/)
