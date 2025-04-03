from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load the trained model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load the soil-to-NPK mapping
soil_npk = pd.read_csv("datasets/soil_npk.csv")
soil_npk_dict = soil_npk.set_index("Soil Type").to_dict(orient="index")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the form
        soil_type = request.form["soil_type"]
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Retrieve N, P, K values based on soil type
        npk_values = soil_npk_dict.get(soil_type, {"N": 50, "P": 20, "K": 30})  # Default to Sandy Soil if unknown
        N, P, K = npk_values["N"], npk_values["P"], npk_values["K"]

        # Create a feature array
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict the crop
        prediction = model.predict(input_data_scaled)[0]

        return render_template("result.html", prediction=prediction)

    # Pass soil types to the template for the dropdown menu
    soil_types = soil_npk["Soil Type"].tolist()
    return render_template("index.html", soil_types=soil_types)

if __name__ == "__main__":
    app.run(debug=False)  # Disable debug mode for production