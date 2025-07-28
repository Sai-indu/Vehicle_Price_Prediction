from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Load the full pipeline (preprocessor + model) using joblib
model_pipeline = joblib.load("vehicle_price_pipeline.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        # ✅ Collect user input from form
        input_data = [
            int(form['year']),
            int(form['cylinders']),
            float(form['mileage']),
            int(form['doors']),
            form['make'],
            form['fuel'],
            form['transmission'],
            form['body'],
            form['drivetrain'],
            form['model'],
            form['trim'],
            form['engine'],
            form['exterior_color'],
            form['interior_color']
        ]

        # ✅ Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data], columns=[
            'year', 'cylinders', 'mileage', 'doors',
            'make', 'fuel', 'transmission', 'body', 'drivetrain',
            'model', 'trim', 'engine', 'exterior_color', 'interior_color'
        ])

        # ✅ Predict using pipeline
        prediction = model_pipeline.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Predicted Price: ₹{int(prediction):,}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
