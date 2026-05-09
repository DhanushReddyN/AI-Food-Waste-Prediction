from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Load trained model
model = joblib.load("models/food_waste_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "Type of Food": [str(request.form["food_type"])],
        "Number of Guests": [int(request.form["guests"])],
        "Event Type": [str(request.form["event_type"])],
        "Quantity of Food": [float(request.form["quantity"])],
        "Storage Conditions": [str(request.form["storage"])],
        "Purchase History": [float(request.form["purchase_history"])],
        "Seasonality": [str(request.form["season"])],
        "Preparation Method": [str(request.form["prep_method"])],
        "Geographical Location": [str(request.form["location"])],
        "Pricing": [str(request.form["pricing"])]
    }

    df = pd.DataFrame(input_data)

    try:
        prediction = model.predict(df)[0]

    except Exception as e:
        return f"Error occurred: {str(e)}"

    recommendation = ""

    if prediction > 35:
        recommendation = (
            "High waste expected. Reduce food preparation quantity."
        )

    elif prediction > 20:
        recommendation = (
            "Moderate waste predicted. Monitor inventory carefully."
        )

    else:
        recommendation = (
            "Low waste expected. Current planning looks efficient."
        )
    insight = ""
    risk = ""
    history = []

    if prediction > 35:
      risk = "HIGH RISK"
      insight = "AI detected high overproduction probability."

    elif prediction > 20:
      risk = "MODERATE RISK"
      insight = "AI suggests monitoring inventory carefully."

    else:
      risk = "LOW RISK"
      insight = "Food planning appears optimized."
    
    history.append(round(prediction,2))

    return render_template(
    "result.html",
    prediction=round(prediction,2),
    recommendation=recommendation,
    risk=risk,
    insight=insight,
    history=history
)


# Run Flask App
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)