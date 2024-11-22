import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Define file paths
test_data_path = os.path.join('data', 'processed', 'test_split.csv')
output_dir = 'output'
tfidf_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
stacked_model_path = os.path.join(output_dir, 'stacked_model.pkl')

# Load pre-trained components
print("Loading TF-IDF vectorizer...")
tfidf_vectorizer = joblib.load(tfidf_path)

print("Loading label encoder...")
label_encoder = joblib.load(label_encoder_path)

print("Loading stacked model...")
stacked_model = joblib.load(stacked_model_path)

# Load and preprocess test data
print("Loading and preprocessing test data...")
test_data = pd.read_csv(test_data_path)
test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].fillna('')

X_test = tfidf_vectorizer.transform(test_data['crimeaditionalinfo'])
y_test = test_data['combined_class']

# Encode test labels, handling unseen labels by filtering them out
print("Filtering unseen labels in test data...")
known_labels = set(label_encoder.classes_)
filtered_test_data = test_data[test_data['combined_class'].isin(known_labels)]
X_test_filtered = tfidf_vectorizer.transform(filtered_test_data['crimeaditionalinfo'])
y_test_filtered = label_encoder.transform(filtered_test_data['combined_class'])

# Evaluate the model
print("Evaluating stacked model...")
y_pred = stacked_model.predict(X_test_filtered)

# Print evaluation metrics
accuracy = accuracy_score(y_test_filtered, y_pred)
classification_rep = classification_report(y_test_filtered, y_pred, target_names=label_encoder.classes_)

print("Evaluation Results:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
