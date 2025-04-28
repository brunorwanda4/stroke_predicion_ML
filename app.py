# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model/stroke_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        user_input = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['Residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status']
        }

        # Encode categorical values
        for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            le = label_encoders[col]
            user_input[col] = le.transform([user_input[col]])[0]

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        result = 'High risk of stroke ⚠️' if prediction == 1 else 'Low risk of stroke ✅'
        return render_template('form.html', result=result)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
