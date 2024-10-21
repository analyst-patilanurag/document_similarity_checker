import streamlit as st
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objs as go

# Streamlit app layout with custom CSS styles
st.markdown(
    """
    <style>
    .text-box {
        background-color: #f0f0f5;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# App title and input section
st.title("Document Similarity Checker")
st.text("This app extracts important words from two documents and compares them using TF-IDF and cosine similarity.")

# Side-by-side text inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    text1 = st.text_area("Enter the first text:", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="text-box">', unsafe_allow_html=True)
    text2 = st.text_area("Enter the second text:", height=150)
    st.markdown('</div>', unsafe_allow_html=True)

# Function to extract top words and their TF-IDF values
def get_top_tfidf_words(text, vectorizer, n=20):
    """Extracts top n words with highest TF-IDF values from the text."""
    tfidf_matrix = vectorizer.transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray().flatten()
    top_indices = tfidf_scores.argsort()[-n:][::-1]  # Get indices of top n scores
    top_words = feature_array[top_indices]
    top_values = tfidf_scores[top_indices]
    return top_words, top_values

# Function to calculate cosine similarity
def calculate_cosine_similarity(text1, text2, vectorizer):
    """Calculates cosine similarity between the two texts."""
    tfidf_matrix = vectorizer.transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity

# Function to perform text analysis
def text_analysis(text, count_vocab):
    """Performs text analysis, calculating total words, stop words removed, and unique words."""
    total_words = len(re.findall(r'\w+', text))
    stop_words_removed = total_words - len(count_vocab)
    unique_words = len(count_vocab)
    return total_words, stop_words_removed, unique_words

# Function to plot word scatter plot using Plotly
def plot_word_scatter(words1, values1, words2, values2):
    fig = go.Figure()

    # Scatter plot for document 1 with blue markers
    fig.add_trace(go.Scatter(
        x=words1, y=values1, mode='markers',
        name='Text 1', text=words1, textposition='top center',
        marker=dict(color='blue', size=10)  # Blue markers
    ))

    # Scatter plot for document 2 with green markers
    fig.add_trace(go.Scatter(
        x=words2, y=values2, mode='markers',
        name='Text 2', text=words2, textposition='top center',
        marker=dict(color='green', size=10)  # Green markers
    ))

    # Update layout
    fig.update_layout(
        title="Word Comparison based on TF-IDF Scores",
        xaxis_title="Words",
        yaxis_title="TF-IDF Score",
        xaxis_tickangle=-45,
        legend_title="Documents"
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# Button to trigger the analysis
if st.button("Check Similarity") and text1.strip() and text2.strip():
    # Initialize the TF-IDF vectorizer and fit on both texts
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit([text1, text2])

    # Get top TF-IDF words for both texts
    words1, values1 = get_top_tfidf_words(text1, vectorizer)
    words2, values2 = get_top_tfidf_words(text2, vectorizer)

    # Calculate cosine similarity between the texts
    similarity = calculate_cosine_similarity(text1, text2, vectorizer)

    # Display similarity result
    st.subheader("Cosine Similarity Result")
    st.write(f"**Cosine Similarity:** {similarity:.4f}")

    if similarity > 0.8:
        st.success("The documents are highly similar.")
    elif similarity > 0.5:
        st.warning("The documents have some similarities.")
    else:
        st.error("The documents are not similar.")

    # Perform text analysis for both documents
    st.subheader("Detailed Text Analysis")
    col1, col2 = st.columns(2)

    with col1:
        total_words1, stop_words_removed1, unique_words1 = text_analysis(text1, vectorizer.get_feature_names_out())
        st.write(f"**Total words in Text 1:** {total_words1}")
        st.write(f"**Stop words removed:** {stop_words_removed1}")
        st.write(f"**Unique words:** {unique_words1}")

    with col2:
        total_words2, stop_words_removed2, unique_words2 = text_analysis(text2, vectorizer.get_feature_names_out())
        st.write(f"**Total words in Text 2:** {total_words2}")
        st.write(f"**Stop words removed:** {stop_words_removed2}")
        st.write(f"**Unique words:** {unique_words2}")

    # Plot the word scatter plot
    st.subheader("TF-IDF Word Comparison Scatter Plot")
    plot_word_scatter(words1, values1, words2, values2)
