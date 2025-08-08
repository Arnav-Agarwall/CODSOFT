import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 1. Load the dataset
# The dataset is loaded, skipping the initial 96 rows and without a header.
df = pd.read_csv("spam.csv", encoding='latin-1', skiprows=96, header=None)

# 2. Preprocessing
# Drop columns that are not needed and rename the remaining ones for clarity.
df = df[[0, 1]]
df.rename(columns={0: 'label', 1: 'text'}, inplace=True)

# Convert the 'label' column to numerical values (ham=0, spam=1).
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Feature Engineering with TF-IDF
# Separate features (X) and target (y).
X = df['text']
y = df['label']

# Initialize TfidfVectorizer to convert text data into a matrix of TF-IDF features.
# It removes common English stop words and considers the top 5000 features.
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# 4. Model Training
# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize and train the LinearSVC model.
model = LinearSVC(random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
# Make predictions on the test set.
predictions = model.predict(X_test)

# Print a detailed classification report.
report = classification_report(y_test, predictions, target_names=['ham', 'spam'])
print(report)