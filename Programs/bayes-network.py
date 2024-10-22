import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load dataset from CSV file
data = pd.read_csv("network.csv", names=['message', 'label'])  # Tabular form data
print('Total instances in the dataset:', data.shape[0])

# Map labels to numeric values
data['label_numeric'] = data.label.map({'pos': 1, 'neg': 0})
X = data.message  # Features (messages)
y = data.label_numeric  # Labels (numeric)

# Display first 5 instances of the messages and their labels
print('\nThe message and its label of first 5 instances are listed below:')
X_sample, y_sample = X[0:5], data.label[0:5]
for message, label in zip(X_sample, y_sample):
    print(f'{message}, {label}')

# Split dataset into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('\nDataset is split into Training and Testing samples')
print('Total training instances:', X_train.shape[0])
print('Total testing instances:', X_test.shape[0])

# Convert text data into numerical format using CountVectorizer
count_vectorizer = CountVectorizer()
X_train_dtm = count_vectorizer.fit_transform(X_train)  # Document-Term Matrix for training data
X_test_dtm = count_vectorizer.transform(X_test)  # Document-Term Matrix for testing data
print('\nTotal features extracted using CountVectorizer:', X_train_dtm.shape[1])

# Display features for first 5 training instances
features_df = pd.DataFrame(X_train_dtm.toarray(), columns=count_vectorizer.get_feature_names_out())
print('\nFeatures for first 5 training instances are listed below:')
print(features_df[0:5])

# Train Naive Bayes classifier
naive_bayes_classifier = MultinomialNB().fit(X_train_dtm, y_train)

# Make predictions on the test data
predictions = naive_bayes_classifier.predict(X_test_dtm)

# Display classification results for testing samples
print('\nClassification results of testing samples are given below:')
for message, prediction in zip(X_test, predictions):
    sentiment = 'pos' if prediction == 1 else 'neg'
    print(f'{message} -> {sentiment}')

# Calculate and display accuracy metrics
print('\nAccuracy metrics:')
print('Accuracy of the classifier is', metrics.accuracy_score(y_test, predictions))
print('Recall:', metrics.recall_score(y_test, predictions))
print('Precision:', metrics.precision_score(y_test, predictions))
print('Confusion matrix:')
print(metrics.confusion_matrix(y_test, predictions))
