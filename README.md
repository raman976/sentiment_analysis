# Sentiment Analysis Research Project

This repository contains a sentiment analysis research project where text data is preprocessed, classical machine learning models are trained, and a transformer-based model (DistilBERT) is fine-tuned for advanced sentiment prediction. The final sentiment is determined using **majority voting** among the models.


The models generated in this repository are used in a separate repository to create a **Streamlit-based Sentiment Analysis Tool**:

## Deployment

- [Sentiment Analysis Tool Repository](https://github.com/raman976/sentiment-analysis-tool)  
- [Live Application](https://mood07.streamlit.app/)

## Project Overview

The project workflow consists of the following steps:

1. **Preprocessing**  
   - Text cleaning, normalization, and tokenization.  
   - Stopword removal, punctuation cleaning.

2. **Model Training**  
   - **Classical Models:** Logistic Regression and Naive Bayes are trained on vectorized data.  
   - **Transformer Model:** DistilBERT is fine-tuned on the dataset.  

3. **Evaluation**  
   - Performance is evaluated using accuracy, precision, recall, and F1-score.  
   - DistilBERT achieves higher accuracy than classical models, as expected.  

4. **Prediction**  
   - Predictions from all three models are combined using **majority voting** to determine the final sentiment.  



