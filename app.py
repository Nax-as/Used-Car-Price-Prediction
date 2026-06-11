"""
Flask Web Application — Used Car Price Prediction
--------------------------------------------------
Routes:
  GET  /          → index.html  (input form)
  POST /predict   → result.html (price estimate)
"""
from flask import Flask, render_template, request
import traceback

from src.pipeline.prediction_pipeline import PredictPipeline, CarData
from src.logger import logger

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Home page with the car detail form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, run prediction, render result."""
    try:
        car = CarData(
            brand        = request.form.get("brand", "").strip(),
            model        = request.form.get("model", "").strip(),
            year         = int(request.form.get("year", 2015)),
            km_driven    = int(request.form.get("km_driven", 50000)),
            fuel         = request.form.get("fuel", "Petrol"),
            seller_type  = request.form.get("seller_type", "Individual"),
            transmission = request.form.get("transmission", "Manual"),
            owner        = request.form.get("owner", "First Owner"),
            mileage      = float(request.form.get("mileage", 18.0)),
            engine       = float(request.form.get("engine", 1200.0)),
            max_power    = float(request.form.get("max_power", 80.0)),
            seats        = int(request.form.get("seats", 5)),
        )

        df      = car.to_dataframe()
        pipeline = PredictPipeline()
        price   = pipeline.predict(df)

        # Format price in Indian numbering (Lakhs / Crores)
        if price >= 10_000_000:
            display = f"₹{price / 10_000_000:.2f} Crore"
        elif price >= 100_000:
            display = f"₹{price / 100_000:.2f} Lakh"
        else:
            display = f"₹{price:,.0f}"

        logger.info(f"Predicted price: {display}")
        return render_template("result.html",
                               price=display,
                               raw_price=f"₹{price:,.2f}",
                               car=car)

    except Exception as e:
        logger.error(traceback.format_exc())
        return render_template("result.html",
                               error=str(e),
                               price=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
