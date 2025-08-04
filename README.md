# 🎬 Movie Genre Predictor

A machine learning project that predicts the **genre** of a movie based on its **plot summary and title**. This is developed as part of the **CODSOFT Machine Learning Internship**.

---

## 📌 Task Objective

> Create a machine learning model that can predict the genre of a movie using textual data like its plot summary. Use NLP techniques like **TF-IDF** or **word embeddings**, and classifiers like **Logistic Regression**, **Naive Bayes**, or **SVM**.

---

## 🧠 Model Details

- **Text Preprocessing**:
  - Lowercasing
  - Removing punctuation and digits
  - Token normalization

- **Vectorization**:
  - TF-IDF Vectorizer (bigrams, stop word removal, max features = 20,000)

- **Classifier**:
  - Logistic Regression wrapped in a One-vs-Rest Classifier
  - Class-weight balanced to handle class imbalance

---
