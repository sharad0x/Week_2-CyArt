import streamlit as st
from nlp_pipeline import *
from utils import *


st.set_page_config(page_title="Text Analytics Web App", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "Data Explorer", "Analysis Dashboard"])

# File upload
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type="txt")
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.raw_text = parse_uploaded_file(uploaded_file)

# Main Page
if page == "Main":
    st.title("NLTK-Powered Text Analytics App")
    st.markdown("""
    Upload a `.txt` file from the sidebar. Use the **Data Explorer** to see tokens, POS tags, 
    sentiment analysis, and the **Analysis Dashboard** to visualize N-grams and trends.
    """)
    if st.session_state.raw_text:
        st.success("File loaded successfully!")

# Data Explorer Page
elif page == "Data Explorer":
    st.title("Data Explorer")

    if st.session_state.raw_text:
        text = st.session_state.raw_text
        tokens = clean_and_tokenize(text)
        tagged = pos_tagging(tokens)
        freq_dist = compute_freq_dist(tokens)
        sentiments = sentiment_scores(text)

        st.subheader("Full Frequency Distribution (Sorted)")

        # Create DataFrame of full frequency distribution
        df_full_freq = pd.DataFrame(freq_dist.items(), columns=["Word", "Frequency"])
        df_full_freq = df_full_freq.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

        # Display full frequency table
        st.dataframe(df_full_freq, use_container_width=True)


        st.subheader("Top 20 Words (Bar Plot)")
        fig_top = plot_top_ngrams(freq_dist, "Top Word Frequencies", top_n=20)
        st.pyplot(fig_top)

        st.subheader("Full Frequency Distribution (Log-Log Plot)")
        fig_full = plot_full_freq_distribution(freq_dist)
        st.pyplot(fig_full)


        st.subheader("POS Tags (First 30 tokens)")
        st.write(pd.DataFrame(tagged[:30], columns=["Token", "POS"]))

        # Collocations
        st.subheader("Collocations (Top Bigrams by PMI)")

        collocations = get_collocations(tokens, top_n=20, min_freq=1)
        df_colloc = pd.DataFrame(collocations)

        if df_colloc.empty:
            st.warning("No collocations found. Try uploading a larger text file or ensure it's properly cleaned.")
        else:
            st.dataframe(df_colloc, use_container_width=True)


        st.subheader("Overall Sentiment Scores")
        st.write(sentiments)
    else:
        st.warning("Please upload a `.txt` file first.")

# Analysis Dashboard Page
elif page == "Analysis Dashboard":
    st.title("Analysis Dashboard")

    if st.session_state.raw_text:
        raw_text = st.session_state.raw_text
        tokens = clean_and_tokenize(raw_text)
        freq = compute_freq_dist(tokens)
        bigram_freq = get_bigrams(tokens)

        # --- N-gram Visualizations ---
        st.subheader("Top Unigrams")
        fig1 = plot_top_ngrams(freq, "Top Unigrams")
        st.pyplot(fig1)

        st.subheader("Top Bigrams(Collocations)")
        fig2 = plot_top_ngrams(bigram_freq, "Top Bigrams")
        st.pyplot(fig2)

        trigram_freq = get_trigrams(tokens)

        st.subheader("Top Trigrams")
        fig3 = plot_top_ngrams(trigram_freq, "Top Trigrams")
        st.pyplot(fig3)

        # --- Sentiment Trend Visualization ---
        st.subheader("Sentiment Trend by Sentence")

        sentences = nltk.sent_tokenize(raw_text)
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(sent)["compound"] for sent in sentences]

        sentiment_df = pd.DataFrame({
            "Sentence #": list(range(1, len(sentences) + 1)),
            "Sentiment Score": scores
        })

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(sentiment_df["Sentence #"], sentiment_df["Sentiment Score"], marker='o', color='purple')
        ax3.axhline(0, color='black', linestyle='--')
        ax3.set_title("Sentiment Trend Over Sentences")
        ax3.set_xlabel("Sentence Number")
        ax3.set_ylabel("Compound Sentiment Score")
        st.pyplot(fig3)

    else:
        st.warning("Please upload a `.txt` file first.")
