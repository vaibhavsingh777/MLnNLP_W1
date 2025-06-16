import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Remove 1 review from each list to make exactly 50
positive_reviews = [
    "Amazing movie!", "I loved the acting.", "Fantastic story.", "Brilliant screenplay.",
    "Absolutely loved it.", "An inspiring movie.", "Emotional and powerful.", "Highly recommend!",
    "Top-notch performance.", "Great plot!", "Kept me hooked.", "Exceptional film.", "Outstanding work.",
    "Beautiful cinematography.", "Marvelous!", "Perfect execution.", "Heartwarming story.", "Loved every second.",
    "Very well made.", "A good movie!", "Instant classic.", "Memorable experience.", "Great acting.",
    "Truly a masterpiece.", "Will watch again.", "Everything perfect.", "Phenomenal film!", "Top-class direction.",
    "Oscar-worthy.", "Brilliant ending.", "Unforgettable ride.", "Great from start.", "Simply superb.",
    "Enthralling story.", "Loved characters.", "Engaging film.", "Beautiful storytelling.", "A cinematic gem.",
    "Masterfully crafted.", "Best movie ever.", "Wonderful acting.", "Visually stunning.", "Flawless script.",
    "Hats off to team.", "Remarkable work.", "Gripping film.", "Heartfelt and moving.",
    "Fantastic in every way.", "Pure brilliance.", "Emotional journey.", "Superb!"
][:50]  # Ensure only 50 items

negative_reviews = [
    "Terrible movie.", "I hated the acting.", "Boring story.", "Weak screenplay.",
    "Absolutely hated it.", "Complete disaster.", "Emotionless and dull.", "Don't recommend!",
    "Awful performance.", "Poor plot!", "Waste of time.", "Mediocre film.", "Very disappointing.",
    "Terrible cinematography.", "Horrible!", "Poor execution.", "Shallow story.", "Hated every second.",
    "Badly made.", "A bad movie!", "Forgettable experience.", "Worst acting.", "Truly a disaster.",
    "Never again.", "Everything awful.", "Horrendous film!", "Poor direction.", "Not worth Oscar.",
    "Terrible ending.", "Regretful ride.", "Bad from start.", "Simply pathetic.", "Dull story.",
    "Weak characters.", "Boring and slow.", "Lazy storytelling.", "A cinematic mess.", "Sloppily crafted.",
    "Worst movie ever.", "Terrible acting.", "Visually ugly.", "Awful script.", "What a waste.",
    "Embarrassing work.", "Sleep-inducing film.", "Heartless and meaningless.", "Bad in every way.",
    "Utter rubbish.", "Painful journey.", "Trash!", "Useless waste of time."
][:50]  # Ensure only 50 items

# SAFETY CHECK
print("Positive Reviews:", len(positive_reviews))
print("Negative Reviews:", len(negative_reviews))

reviews = pd.DataFrame({
    'Review': positive_reviews + negative_reviews,
    'Sentiment': ['positive'] * 50 + ['negative'] * 50
})

vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(reviews['Review'])
y = reviews['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    return model.predict(review_vector)[0]

# Example predictions
example_review_1 = "I absolutely loved the movie, it was brilliant!"
example_review_2 = "This was a horrible waste of time."

print("\n✅ Example Prediction 1:", predict_review_sentiment(model, vectorizer, example_review_1))
print("✅ Example Prediction 2:", predict_review_sentiment(model, vectorizer, example_review_2))
