# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess data
data = pd.read_csv('data/data.csv', na_values=['N/A'])
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=['id', 'stroke'])
y = data['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model/stroke_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')

print("Model and encoders saved!")
