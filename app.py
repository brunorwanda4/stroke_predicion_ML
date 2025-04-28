# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd
from pymongo import MongoClient, DESCENDING # Import DESCENDING for sorting
import datetime

app = Flask(__name__)

# --- MongoDB Setup ---
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['stroke_prediction_db']
    predictions_collection = db['user_inputs']
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Could not connect to MongoDB: {e}")
    predictions_collection = None
# --- End MongoDB Setup ---

# --- Load model and encoders ---
try:
    model = joblib.load('model/stroke_model.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
except FileNotFoundError:
    print("Error: Model or encoder files not found. Make sure 'model/stroke_model.pkl' and 'model/label_encoders.pkl' exist.")
    exit()
# --- End Load model ---

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    history_data = [] # Initialize empty list for history

    # --- Fetch History Data ---
    # Fetch data regardless of GET or POST, before rendering
    if predictions_collection is not None:
        try:
            # Fetch latest 10 records, sorted by timestamp descending
            # Use .find() without arguments to get all fields
            history_cursor = predictions_collection.find().sort('timestamp', DESCENDING).limit(10)
            # Convert cursor to list of dictionaries to pass to template
            history_data = list(history_cursor)
            # Note: _id objects are kept as is; Jinja can handle them,
            # but if needed elsewhere, convert using str(record['_id'])
        except Exception as e:
            print(f"Error fetching history from MongoDB: {e}")
            # Keep history_data as empty list on error
    else:
        print("MongoDB collection not available. Cannot fetch history.")
    # --- End Fetch History Data ---


    if request.method == 'POST':
        try:
            # Get user input (store original values for DB)
            user_input_original = {
                'gender': request.form['gender'],
                'age': float(request.form['age']),
                'hypertension': 1 if request.form.get('hypertension') == 'on' else 0,
                'heart_disease': 1 if request.form.get('heart_disease') == 'on' else 0,
                'ever_married': request.form['ever_married'],
                'work_type': request.form['work_type'],
                'Residence_type': request.form['Residence_type'],
                'avg_glucose_level': float(request.form['avg_glucose_level']),
                'bmi': float(request.form['bmi']),
                'smoking_status': request.form['smoking_status'],
                'timestamp': datetime.datetime.now(datetime.timezone.utc)
            }

            # Prepare data for prediction (encoding)
            user_input_encoded = user_input_original.copy()
            del user_input_encoded['timestamp']

            for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
                if col in label_encoders:
                    le = label_encoders[col]
                    if user_input_encoded[col] in le.classes_:
                        user_input_encoded[col] = le.transform([user_input_encoded[col]])[0]
                    else:
                        raise ValueError(f"Unknown category '{user_input_encoded[col]}' for column '{col}'")
                else:
                     raise KeyError(f"Label encoder not found for column '{col}'")

            input_df = pd.DataFrame([user_input_encoded])
            prediction = model.predict(input_df)[0]
            result = 'High risk of stroke ⚠️' if prediction == 1 else 'Low risk of stroke ✅'

            # --- Save data to MongoDB ---
            if predictions_collection is not None:
                try:
                    document_to_save = user_input_original.copy()
                    document_to_save['prediction_result_code'] = int(prediction)
                    document_to_save['prediction_result_text'] = result
                    insert_result = predictions_collection.insert_one(document_to_save)
                    print(f"Data saved to MongoDB with ID: {insert_result.inserted_id}")
                    # *** Refresh history after saving ***
                    # Re-fetch history to include the latest submission immediately
                    history_cursor = predictions_collection.find().sort('timestamp', DESCENDING).limit(10)
                    history_data = list(history_cursor)
                except Exception as e:
                    print(f"Error saving data to MongoDB: {e}")
            else:
                print("MongoDB collection not available. Skipping save.")
            # --- End Save ---

        except ValueError as ve:
             print(f"Data processing error: {ve}")
             result = f"Error in input data: {ve}"
        except KeyError as ke:
            print(f"Missing data error: {ke}")
            result = f"Missing input field or configuration error: {ke}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            result = "An unexpected error occurred during prediction."

        # Render template after POST, passing both result and history
        return render_template('form.html', result=result, history=history_data)

    # For GET requests, render template passing only the history
    return render_template('form.html', history=history_data) # No result on initial GET

if __name__ == '__main__':
    app.run(debug=True)