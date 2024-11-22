import pandas as pd
from sklearn.model_selection import train_test_split
import os

# File paths
input_file = 'data/train.csv'
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading data...")
data = pd.read_csv(input_file)

# Fill missing values
print("Handling missing values...")
data['category'] = data['category'].fillna('Unknown')
data['sub_category'] = data['sub_category'].fillna('Unknown')
data['crimeaditionalinfo'] = data['crimeaditionalinfo'].fillna('')

# Combine category and sub_category for stratification
print("Creating combined class column...")
data['combined_class'] = data['category'] + " | " + data['sub_category']

# Handle rare classes (threshold set to minimum 5 instances)
print("Handling rare classes...")
class_counts = data['combined_class'].value_counts()
rare_classes = class_counts[class_counts < 5].index
data['combined_class'] = data['combined_class'].apply(lambda x: 'Other' if x in rare_classes else x)

# Drop rows with the "Other" class (optional, to ensure stratified splitting works)
if 'Other' in data['combined_class'].unique():
    print("Dropping rows with 'Other' class...")
    data = data[data['combined_class'] != 'Other']

# Ensure sufficient instances in all classes
print("Class distribution after handling rare classes:")
print(data['combined_class'].value_counts())

# Stratified split
print("Splitting data into Train (60%), Test (20%), and Validation (20%) sets...")
train_data, temp_data = train_test_split(
    data, test_size=0.4, random_state=42, stratify=data['combined_class']
)
test_data, val_data = train_test_split(
    temp_data, test_size=0.5, random_state=42, stratify=temp_data['combined_class']
)

# Save splits with the combined_class column included for better debugging and use
print("Saving data splits...")
train_data.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
val_data.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)

print("Data split complete:")
print(f"Train data saved to {os.path.join(output_dir, 'train_split.csv')} ({len(train_data)} rows)")
print(f"Test data saved to {os.path.join(output_dir, 'test_split.csv')} ({len(test_data)} rows)")
print(f"Validation data saved to {os.path.join(output_dir, 'val_split.csv')} ({len(val_data)} rows)")
