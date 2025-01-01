import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load dataset
msg = pd.read_csv("Program6dataset.csv", names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(msg.message, msg.labelnum, test_size=0.3, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)
# Displaying features for the first 5 training instances
print('\nFeatures for first 5 training instances are listed below')
df = pd.DataFrame(X_train_dtm.toarray(), columns=vectorizer.get_feature_names_out())
print(df.head())
# Train model
model = MultinomialNB().fit(X_train_dtm, y_train)

# Predict and evaluate
predictions = model.predict(X_test_dtm)
print("Accuracy:", metrics.accuracy_score(y_test, predictions))
print("Recall:", metrics.recall_score(y_test, predictions))
print("Precision:", metrics.precision_score(y_test, predictions))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, predictions))
