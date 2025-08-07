import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist, pos_tag, bigrams
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import trigrams

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def clean_and_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalpha()]
    
    return tokens

def pos_tagging(tokens):
    return pos_tag(tokens)

def compute_freq_dist(tokens):
    return FreqDist(tokens)

def get_bigrams(tokens):
    return FreqDist(bigrams(tokens))

def get_trigrams(tokens):
    return FreqDist(trigrams(tokens))


def sentiment_scores(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)


def get_collocations(tokens, top_n=20, min_freq=1):
    if not tokens or len(tokens) < 10:
        return []
    
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(min_freq)

    scored = finder.score_ngrams(BigramAssocMeasures.pmi)

    # Sort and get top-N collocations
    top_collocations = sorted(scored, key=lambda x: -x[1])[:top_n]
    
    # Return as list of tuples
    return [{"Bigram": f"{w1} {w2}", "PMI": round(pmi, 3)} for (w1, w2), pmi in top_collocations]
