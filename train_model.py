# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os # Import os module

# --- Configuration ---
DATA_PATH = 'data/data.csv'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'stroke_model.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')

# --- Ensure model directory exists ---
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Ensured model directory exists: {MODEL_DIR}")

# --- Load Data ---
print(f"Loading data from: {DATA_PATH}")
try:
    # Treat 'N/A' and potentially empty strings as missing values
    data = pd.read_csv(DATA_PATH, na_values=['N/A', ''])
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

print("Data loaded successfully. Initial shape:", data.shape)
# print("Initial missing values:\n", data.isnull().sum()) # Optional: Check initial NaNs

# --- Preprocessing ---
# Handle missing BMI (Mean Imputation)
if 'bmi' in data.columns:
    if data['bmi'].isnull().any():
        bmi_mean = data['bmi'].mean()
        data['bmi'] = data['bmi'].fillna(bmi_mean)
        print(f"Filled missing BMI values with mean: {bmi_mean:.2f}")
    else:
        print("No missing BMI values found.")
else:
    print("Warning: 'bmi' column not found in data.")

# Identify categorical columns (ensure these match your data)
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

print("Applying Label Encoding...")
for col in categorical_cols:
    if col in data.columns:
        # Optional: Handle potential missing values in categorical columns before encoding
        # data[col] = data[col].fillna('Unknown') # Or another appropriate placeholder

        le = LabelEncoder()
        try:
            data[col] = le.fit_transform(data[col].astype(str)) # Ensure consistent type
            label_encoders[col] = le
            print(f" - Encoded '{col}'. Classes: {le.classes_}")
        except Exception as e:
            print(f"Error encoding column '{col}': {e}")
            # Decide how to handle: skip column, exit, etc.
    else:
         print(f"Warning: Categorical column '{col}' not found in data.")

# Check for any remaining missing values after preprocessing
# print("Missing values after preprocessing:\n", data.isnull().sum())

# --- Feature and Target Split ---
print("Splitting data into features (X) and target (y)...")
try:
    # Ensure 'id' and 'stroke' columns exist before dropping/selecting
    if 'id' in data.columns:
        X = data.drop(columns=['id', 'stroke'])
    else:
         X = data.drop(columns=['stroke']) # If no 'id' column
    y = data['stroke']
except KeyError as e:
    print(f"Error: Missing expected column for split: {e}")
    exit()

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# --- Train/Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Use stratify for imbalanced data
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Model Training ---
print("Training RandomForestClassifier model...")
# Using class_weight='balanced' is good for imbalance
model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100) # n_estimators is a common parameter to set
model.fit(X_train, y_train)
print("Model training complete.")

# --- Evaluation ---
print("\n--- Model Evaluation ---")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nTraining Set Performance:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
# print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
# print("Classification Report:\n", classification_report(y_train, y_pred_train))

print("\nTesting Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
print("------------------------\n")

# --- Save Model and Encoders ---
print(f"Saving model to: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"Saving label encoders to: {ENCODERS_PATH}")
joblib.dump(label_encoders, ENCODERS_PATH)

print("\nModel and encoders saved successfully!")