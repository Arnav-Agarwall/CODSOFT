Spam SMS Detection
This project implements an Artificial Intelligence model to classify SMS messages as either spam or legitimate (ham). The primary goal is to provide an effective solution for filtering unwanted messages, enhancing user experience and security.

Features
Data Preprocessing: Handles raw SMS text data, including cleaning and preparing it for model training.

Feature Engineering: Utilizes CountVectorizer to convert text messages into numerical feature vectors, making them suitable for machine learning algorithms.

Machine Learning Model: Employs a Multinomial Naive Bayes classifier, which is well-suited for text classification tasks.

Model Evaluation: Provides detailed performance metrics, including precision, recall, and F1-score, to assess the model's effectiveness.

Model Performance

The current iteration of the model, using CountVectorizer and Multinomial Naive Bayes, exhibits strong performance:

Accuracy: The model correctly classifies 97% of all messages overall.

Spam Precision: Out of all messages predicted as spam, 99% are actually spam.

Spam Recall: The model correctly identifies 88% of all actual spam messages.
