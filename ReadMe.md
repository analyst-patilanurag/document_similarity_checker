# **Multi-Model Text Summarization with Vectorization Insights**

This Streamlit app allows users to generate and compare summaries using multiple NLP models and visualize word vectorization insights. It integrates **T5**, **Pegasus**, and **DistilBART** models for summarization and provides detailed text analysis through **CountVectorizer** and **TfidfVectorizer**.

---

## **Features**
1. **Multi-Model Summarization:**
   - Generate summaries using **T5**, **Pegasus**, and **DistilBART**.
   - Compare summaries across models.

2. **Vectorization-Based Word Analysis:**
   - Visualize the top 10 words based on **word frequency** (CountVectorizer) and **TF-IDF score** (TfidfVectorizer).
   - Interactive bar plots to display the results.

3. **Detailed Text Analysis:**
   - Compare the original text and the summarized versions.
   - Analyze **stop words removal** and unique words from the input text.

4. **Highlighted Summaries:**
   - Top words from vectorization are highlighted in the generated summaries.

5. **Comparison of Highlighted Words:**
   - Evaluate how many of the top 10 words are used by each summarization model.

---

## **Technologies Used**
- **Streamlit:** Web framework for the interactive user interface.
- **Transformers Library:** Summarization models including **T5**, **Pegasus**, and **DistilBART**.
- **scikit-learn:** Vectorization tools (**CountVectorizer**, **TfidfVectorizer**).
- **Matplotlib:** Visualization of word rankings.

---

## **How to Run the App Locally**

### Prerequisites
Make sure you have Python installed on your system. Then, install the required dependencies:

```bash
pip install streamlit matplotlib scikit-learn transformers
```

### Run the App
1. Save the Python code as `app.py`.
2. Open a terminal in the directory where `app.py` is located.
3. Run the following command:

```bash
streamlit run app.py
```

4. The app will open in your browser at `http://localhost:8501`.

---

## **App Layout and Workflow**

1. **Input Section:**  
   - Paste the text you want to summarize in the provided text box.

2. **Summarization Models:**  
   - Generate summaries by clicking the **Analyze Text and Generate Summaries** button.
   - The summaries from **T5**, **Pegasus**, and **DistilBART** will be displayed with highlighted key words.

3. **Detailed Text Analysis:**  
   - See the total words, words in the summary, stop words removed, and unique words after stop word removal.

4. **Comparison of Highlighted Words:**  
   - Check how many top words from vectorization are used in each summary.

5. **Word Analysis Visualization:**  
   - View the top 10 words by **word frequency** and **TF-IDF score** using interactive bar charts.

---

## **Screenshots (Optional)**
- **Summarization Results:** Display multiple summaries with highlighted top words.
- **Detailed Text Analysis:** Comparison of word statistics between the input and summaries.
- **Word Ranking Plots:** Interactive bar charts for word frequency and TF-IDF scores.

---

## **Customization Options**

1. **CSS Styling:**
   - The application includes custom CSS to style different sections with padding, rounded corners, and color themes.

2. **Model Adjustments:**
   - The summarization models used (T5, Pegasus, DistilBART) can be replaced with other models from the `transformers` library.

3. **Height Adjustment:**
   - Adjust the height of the input text area by modifying `height=200` in the `st.text_area()` function.

---

## **Known Issues**

- **Performance:** Large input texts may take longer to process due to the summarization models.
- **Dependencies:** Ensure all dependencies are installed correctly to avoid import errors.

---

## **Contributing**

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Contact**

If you have any questions or issues, feel free to reach out.

---

This README file provides all the necessary details to set up, run, and understand the functionalities of your Streamlit app. Let me know if you need further customization!
