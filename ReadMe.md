# Document Similarity Checker - README

## Overview

The **Document Similarity Checker** is a Streamlit application that compares two input texts by extracting key words using **TF-IDF** and calculating **cosine similarity** between them. This app provides an interactive analysis with a scatter plot visualization of the top TF-IDF words from both texts using **Plotly**. It helps users understand how similar two documents are and offers insights into word distributions and TF-IDF scores.

## Features

- **Text Input Fields:** Allows users to input two documents for comparison.
- **Cosine Similarity Calculation:** Measures the similarity between the two input texts.
- **TF-IDF Word Extraction:** Displays the top words with the highest TF-IDF scores for each document.
- **Interactive Scatter Plot:** Visualizes TF-IDF scores for the top words using Plotly with blue and green markers.
- **Detailed Text Analysis:** Provides insights on the total words, stop words removed, and unique words in both documents.

---

## Installation

1. **Clone the repository** or copy the code into your local project.
2. **Install dependencies:**

   Run the following command to install the required packages:
   ```bash
   pip install streamlit scikit-learn plotly
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## Code Explanation

1. **Libraries Used:**
   - `streamlit`: For creating the web interface.
   - `numpy`: For numerical operations.
   - `re`: For text analysis using regular expressions.
   - `scikit-learn`: For implementing **TF-IDF Vectorizer** and **cosine similarity**.
   - `plotly`: For creating interactive scatter plots.

2. **Text Input:**  
   The app allows users to enter two pieces of text for comparison using Streamlit's `text_area`.

3. **TF-IDF Calculation:**  
   It extracts the top 20 words with the highest **TF-IDF scores** from each text using `TfidfVectorizer`.

4. **Cosine Similarity:**  
   The cosine similarity score is computed to quantify the similarity between the two documents.

5. **Plotly Scatter Plot:**  
   An interactive scatter plot is generated with **blue markers** for Text 1 and **green markers** for Text 2. It helps visualize the TF-IDF scores for the top words from both texts.

6. **Text Analysis:**  
   The app provides a detailed analysis of the total words, stop words removed, and unique words in both texts.

---

## How to Use

1. **Enter Text:**  
   Enter two different pieces of text in the provided input fields.

2. **Click "Check Similarity":**  
   Press the button to perform the analysis.

3. **View Results:**
   - **Cosine Similarity Score:** Displays how similar the two documents are.
   - **Detailed Text Analysis:** Shows word statistics for both texts.
   - **Scatter Plot:** Provides a visual comparison of the top TF-IDF words.

---

## Example Usage

1. **Input Texts:**
   - Text 1: "The quick brown fox jumps over the lazy dog."
   - Text 2: "The fast fox leaped over the sleepy hound."

2. **Cosine Similarity Score:**  
   Displays a similarity score (e.g., `0.1016`).

3. **Scatter Plot:**  
   An interactive plot with TF-IDF word scores, where Text 1 is represented with **blue markers** and Text 2 with **green markers**.

---

## Dependencies

- Python 3.7+
- Streamlit
- scikit-learn
- Plotly

---

## Future Improvements

- Add support for **more advanced similarity algorithms** (e.g., Jaccard similarity).
- Provide **stopword customization** options for users.
- Save and export the results as **PDF or CSV reports**.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or feedback, please contact [Anurag Patil](https://www.linkedin.com).