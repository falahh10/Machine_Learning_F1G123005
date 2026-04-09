from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        data = [float(x) for x in request.form.values()]
        
        # Ubah ke array
        data = np.array(data).reshape(1, -1)
        
        # Scaling
        data = scaler.transform(data)
        
        # Prediksi
        prediction = model.predict(data)

        # Hasil
        if prediction[0] == 1:
            result = "🚨 TRANSAKSI FRAUD"
        else:
            result = "✅ TRANSAKSI AMAN"

        return render_template('dashboard.html', result=result)

    except Exception as e:
        return f"Terjadi error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
