import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib  # For saving the model/vectorizer

# ---------------------------- #
# Step 1: Generate Synthetic Dataset
# ---------------------------- #

positive_feedback = [
    "Excellent quality", "Loved the product", "Highly recommended", "Amazing experience",
    "Would buy again", "Top notch service", "Totally satisfied", "Very happy", "Exceeded expectations",
    "Value for money", "Great product", "Superb build", "Wonderful experience", "Terrific purchase",
    "Fantastic support", "Perfect fit", "Affordable and reliable", "Loved the packaging",
    "Efficient and effective", "Reliable product", "User friendly", "Fast delivery",
    "Very pleased", "Just awesome", "Absolutely brilliant"
]

negative_feedback = [
    "Poor quality", "Not recommended", "Bad experience", "Waste of money",
    "Very disappointed", "Terrible product", "Do not buy", "Worst purchase ever",
    "Faulty packaging", "Low durability", "Broken item", "Delayed delivery",
    "Doesn't match description", "Completely useless", "Horrible support",
    "Feels cheap", "Bad build quality", "Unsatisfactory performance",
    "Too expensive for quality", "Never buying again", "Flimsy material",
    "Not worth it", "Doesn't work properly", "Frustrating experience",
    "Extremely dissatisfied"
]

positive_samples = random.choices(positive_feedback, k=50)
negative_samples = random.choices(negative_feedback, k=50)

df = pd.DataFrame({
    'Text': positive_samples + negative_samples,
    'Label': ['good'] * 50 + ['bad'] * 50
}).sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------- #
# Step 2: Text Preprocessing with TF-IDF
# ---------------------------- #

vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')

X = df['Text']
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------- #
# Step 3: Hyperparameter Tuning with GridSearchCV
# ---------------------------- #

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # L2 regularization is standard for Logistic Regression
    'solver': ['liblinear']  # Good for small datasets
}

logreg = LogisticRegression(random_state=42)

grid_search = GridSearchCV(
    logreg, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1_macro'
)
grid_search.fit(X_train_tfidf, y_train)

best_model = grid_search.best_estimator_

print("Best Parameters Found:", grid_search.best_params_)

# ---------------------------- #
# Step 4: Evaluation
# ---------------------------- #

y_pred = best_model.predict(X_test_tfidf)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Precision:", round(precision_score(y_test, y_pred, pos_label='good'), 3))
print("Recall:", round(recall_score(y_test, y_pred, pos_label='good'), 3))
print("F1-Score:", round(f1_score(y_test, y_pred, pos_label='good'), 3))

# ---------------------------- #
# Step 5: Saving Model and Vectorizer
# ---------------------------- #

joblib.dump(best_model, 'logistic_regression_feedback_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer_feedback.joblib')

print("✅ Model and Vectorizer saved successfully.")

# ---------------------------- #
# Step 6: Utility Function for Vectorization + Prediction
# ---------------------------- #

def text_preprocess_vectorize(texts, fitted_vectorizer):
    """
    Takes a list of text samples and a fitted TfidfVectorizer, 
    returns the corresponding TF-IDF feature matrix.
    """
    return fitted_vectorizer.transform(texts)

def predict_feedback_sentiment(model, vectorizer, review):
    """
    Predict sentiment for a single review using the trained model and fitted vectorizer.
    """
    vector = text_preprocess_vectorize([review], vectorizer)
    return model.predict(vector)[0]

# Example usage
loaded_model = joblib.load('logistic_regression_feedback_model.joblib')
loaded_vectorizer = joblib.load('tfidf_vectorizer_feedback.joblib')

example = "Absolutely brilliant product, exceeded my expectations"
print(f"Predicted Sentiment for '{example}':", predict_feedback_sentiment(loaded_model, loaded_vectorizer, example))

