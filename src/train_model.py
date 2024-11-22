import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Define file paths
train_data_path = os.path.join('data', 'processed', 'train_split.csv')
test_data_path = os.path.join('data', 'processed', 'test_split.csv')
val_data_path = os.path.join('data', 'processed', 'val_split.csv')
output_dir = 'output'
tfidf_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
stacked_model_path = os.path.join(output_dir, 'stacked_model.pkl')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess training data
print("Loading and preprocessing training data...")
train_data = pd.read_csv(train_data_path)
train_data['crimeaditionalinfo'] = train_data['crimeaditionalinfo'].fillna('')

# Apply TF-IDF vectorization
print("Applying TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, max_df=0.9, min_df=5)
X_train = tfidf_vectorizer.fit_transform(train_data['crimeaditionalinfo'])
y_train = train_data['combined_class']

# Save TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, tfidf_path)
print(f"TF-IDF vectorizer saved to {tfidf_path}")

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Save label encoder
joblib.dump(label_encoder, label_encoder_path)
print(f"Label encoder saved to {label_encoder_path}")

# Load validation data
print("Loading validation data...")
val_data = pd.read_csv(val_data_path)
val_data['crimeaditionalinfo'] = val_data['crimeaditionalinfo'].fillna('')
X_val = tfidf_vectorizer.transform(val_data['crimeaditionalinfo'])
y_val = label_encoder.transform(val_data['combined_class'])

# Define base models with optimized parameters
print("Setting up base models for stacking...")
random_forest = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced')
gradient_boosting = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
xgboost = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
    max_depth=8,
    learning_rate=0.1,
    n_estimators=150,
    min_child_weight=3,
    scale_pos_weight=1
)

# Meta-model
meta_model = LogisticRegression(max_iter=2000, random_state=42)

# Create the stacking classifier
print("Creating stacking model with RandomForest, GradientBoosting, and XGBoost...")
stacked_model = StackingClassifier(
    estimators=[
        ('random_forest', random_forest),
        ('gradient_boosting', gradient_boosting),
        ('xgboost', xgboost)
    ],
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

# Train the stacked model
print("Training stacked model...")
stacked_model.fit(X_train, y_train_encoded)

# Save the stacked model
joblib.dump(stacked_model, stacked_model_path)
print(f"Stacked model trained and saved to {stacked_model_path}")

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

# Decode predictions for readable results
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Print evaluation metrics
accuracy = accuracy_score(y_test_filtered, y_pred)
classification_rep = classification_report(y_test_filtered, y_pred_decoded, target_names=label_encoder.classes_)

print("Evaluation Results:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
