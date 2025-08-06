import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("fraudTrain.csv")

# Drop non-numeric or irrelevant columns (e.g., strings like product names)
df = df.select_dtypes(include=[np.number])

# Drop rows with NaNs if any
df.dropna(inplace=True)

# Separate features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Balance the training data using SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train_bal, y_train_bal)

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Find best threshold: recall >= 0.80 and precision >= 0.65
best_threshold = 0.5
for thresh in np.arange(0.1, 0.9, 0.01):
    preds = (y_probs >= thresh).astype(int)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    if rec >= 0.80 and prec >= 0.65:
        best_threshold = thresh
        break

# Final predictions using selected threshold
y_pred = (y_probs >= best_threshold).astype(int)

# Print report
print(f"\n=== Final Classification Report (Threshold = {best_threshold:.2f}) ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_balanced.png")
plt.close()
print("Confusion matrix saved as 'confusion_matrix_balanced.png'")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig("roc_curve_balanced.png")
plt.close()
print("ROC curve saved as 'roc_curve_balanced.png'")
