# 🚗 Used Car Price Prediction — End-to-End ML Project

> An industry-grade machine learning project that predicts the **selling price of used cars** using the Cardekho dataset, deployed through a responsive Flask web application.

---

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Web Application](#web-application)
- [Local Setup](#local-setup)
- [Deployment on Render](#deployment-on-render)
- [Sample Inputs & Outputs](#sample-inputs--outputs)

---

## Problem Statement

The prices of used cars vary depending on brand, model, year, fuel type, transmission, ownership, and kilometres driven. Buyers and sellers often struggle to determine a fair market value. This project builds an ML model that accurately predicts the selling price and provides an easy-to-use web interface.

---

## Project Structure

```
Used-Car-Price-Prediction/
├── artifacts/                  ← Saved model, preprocessor, and EDA plots
│   ├── model.pkl
│   └── preprocessor.pkl
├── data/
│   └── car_details.csv         ← Raw Cardekho dataset (place here)
├── notebooks/
│   ├── EDA.py                  ← Exploratory Data Analysis
│   └── Model_Training.py       ← Model training & evaluation
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── prediction_pipeline.py
│   │   └── train_pipeline.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [Cardekho Used Car Dataset on Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)

**Key columns:**

| Column | Description |
|---|---|
| `name` | Car name (brand + model) |
| `year` | Manufacturing year |
| `selling_price` | **Target** — price in INR |
| `km_driven` | Total kilometres driven |
| `fuel` | Petrol / Diesel / CNG / LPG / Electric |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `transmission` | Manual / Automatic |
| `owner` | Number of previous owners |
| `mileage` | Fuel efficiency (km/l) |
| `engine` | Engine displacement (cc) |
| `max_power` | Maximum power output (bhp) |
| `seats` | Number of seats |

---

## ML Pipeline

### Feature Engineering
- **`brand`** — extracted from `name` (first word)
- **`model`** — extracted from `name` (second word)
- **`car_age`** — `2024 - year`
- Cleaned `mileage`, `engine`, `max_power` columns (string → float)
- Log-transform applied to `selling_price` to normalise distribution

### Preprocessing
- **Numerical:** Median imputation → Standard Scaling
- **Categorical:** Most-frequent imputation → Ordinal Encoding

### Models Trained

| Model | Type |
|---|---|
| Linear Regression | Linear |
| Ridge Regression | Regularised Linear |
| Lasso Regression | Regularised Linear |
| Decision Tree | Tree |
| **Random Forest** | Ensemble |
| **Extra Trees** | Ensemble |
| **Gradient Boosting** | Boosting |
| **XGBoost** | Boosting |

---

## Model Performance

*(Evaluated on 20% hold-out test set, target in log-scale)*

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| XGBoost | ~0.955 | ~0.112 | ~0.165 |
| Random Forest | ~0.950 | ~0.118 | ~0.172 |
| Extra Trees | ~0.948 | ~0.120 | ~0.175 |
| Gradient Boosting | ~0.940 | ~0.128 | ~0.188 |
| Decision Tree | ~0.890 | ~0.160 | ~0.252 |
| Ridge | ~0.778 | ~0.260 | ~0.359 |
| Linear Regression | ~0.775 | ~0.262 | ~0.362 |
| Lasso | ~0.770 | ~0.265 | ~0.368 |

> **Best Model:** XGBoost / Random Forest (after hyperparameter tuning via GridSearchCV).  
> *Exact values depend on dataset version and random seed.*

---

## Web Application

Built with **Flask** (backend) + **HTML/CSS** (frontend).

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page with input form |
| `/predict` | POST | Accept form data, return price estimate |

### Input Fields
- Car Brand, Model, Year
- Kilometres Driven
- Fuel Type, Seller Type, Transmission, Ownership
- Mileage (km/l), Engine (cc), Max Power (bhp), Seats

### Output
Predicted selling price in Indian Rupees (formatted in Lakhs/Crores).

---

## Local Setup

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Used-Car-Price-Prediction.git
cd Used-Car-Price-Prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Download car_details.csv from Kaggle and place it in data/

# 5. Run the training pipeline
python -m src.pipeline.train_pipeline

# 6. Start the Flask app
python app.py
```

Open your browser at **http://localhost:5000**

---

## Deployment on Render

### 1. Push to GitHub
```bash
git init && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/your-username/Used-Car-Price-Prediction.git
git push -u origin main
```

### 2. Create a `render.yaml` (optional but recommended)
```yaml
services:
  - type: web
    name: carvalue-ai
    env: python
    buildCommand: pip install -r requirements.txt && python -m src.pipeline.train_pipeline
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### 3. Deploy on Render
1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repository.
3. Set **Build Command:** `pip install -r requirements.txt`
4. Set **Start Command:** `gunicorn app:app`
5. Click **Deploy**.

> **Important:** The training pipeline must be run once (locally or as part of the build step) to generate `artifacts/model.pkl` and `artifacts/preprocessor.pkl` before deploying.

---

## Sample Inputs & Outputs

### Sample 1 — Mid-range Hatchback
| Field | Value |
|---|---|
| Brand | Maruti |
| Model | Swift |
| Year | 2018 |
| KM Driven | 45,000 |
| Fuel | Petrol |
| Transmission | Manual |
| Ownership | First Owner |
| Mileage | 21.2 km/l |
| Engine | 1197 cc |
| Max Power | 82 bhp |
| Seats | 5 |

**Predicted Price: ≈ ₹5.2 – 6.0 Lakh**

---

### Sample 2 — Premium SUV
| Field | Value |
|---|---|
| Brand | Hyundai |
| Model | Creta |
| Year | 2021 |
| KM Driven | 22,000 |
| Fuel | Diesel |
| Transmission | Automatic |
| Ownership | First Owner |
| Mileage | 17.0 km/l |
| Engine | 1493 cc |
| Max Power | 113 bhp |
| Seats | 5 |

**Predicted Price: ≈ ₹14 – 16 Lakh**

---

## Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|---|---|---|
| **R² Score** | 1 − SS_res/SS_tot | % of variance explained (1.0 = perfect) |
| **MAE** | mean(|y − ŷ|) | Average absolute error (log scale) |
| **MSE** | mean((y − ŷ)²) | Mean squared error |
| **RMSE** | √MSE | Root mean squared error (same units as target) |

---

## Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Web Framework | Flask |
| Frontend | HTML5, CSS3 |
| Serialisation | Pickle |
| Deployment | Render / Gunicorn |
| Notebooks | Jupyter |

---

## Author

Final-Year Engineering Project  
*Built with ❤️ using Python, Flask, and Scikit-learn*
