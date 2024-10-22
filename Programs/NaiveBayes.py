# Import necessary libraries
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('NB.csv')
print("The first 5 rows of the dataset:\n", data.head())

# Split the dataset into features (X) and target (y)
features = data.iloc[:, :-1]  # All columns except the last one
target = data.iloc[:, -1]     # The last column

print("\nThe first 5 rows of the features (X):\n", features.head())
print("\nThe first 5 rows of the target (y):\n", target.head())

# Initialize LabelEncoder to convert categorical data into numerical form
label_encoder_outlook = LabelEncoder()
label_encoder_temperature = LabelEncoder()
label_encoder_humidity = LabelEncoder()
label_encoder_wind = LabelEncoder()

# Apply LabelEncoder to each feature column to transform categorical values to numerical values
features['Outlook'] = label_encoder_outlook.fit_transform(features['Outlook'])
features['Temperature'] = label_encoder_temperature.fit_transform(features['Temperature'])
features['Humidity'] = label_encoder_humidity.fit_transform(features['Humidity'])
features['Wind'] = label_encoder_wind.fit_transform(features['Wind'])

print("\nFeatures after Label Encoding (numerical representation):\n", features.head())

# Encode the target column (PlayTennis) into numerical format
label_encoder_play_tennis = LabelEncoder()
target = label_encoder_play_tennis.fit_transform(target)

print("\nTarget after Label Encoding (numerical representation):\n", target)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20)

# Initialize the Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Train the Naive Bayes model on the training data
naive_bayes_classifier.fit(X_train, y_train)

# Predict the test data and calculate accuracy
predictions = naive_bayes_classifier.predict(X_test)
accuracy = accuracy_score(predictions, y_test)

print("Model Accuracy:", accuracy)
