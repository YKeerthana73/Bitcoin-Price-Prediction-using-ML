# 📈 Bitcoin Price Prediction Using Machine Learning

This project is a Machine Learning-powered web application built with **Streamlit** that predicts future Bitcoin prices using historical data. It uses various ML models (Linear Regression, Random Forest) and technical indicators (EMA, MACD, RSI, PSAR) to generate predictions.

## 📌 Features

- 📊 Visualizes historical Bitcoin price trends  
- 🔁 Choose prediction range (e.g., 30–365 days)  
- 📉 Trains multiple models:  
  - Linear Regression  
  - Random Forest  
  - *(Optional: LSTM, if implemented)*  
- 💹 Uses technical indicators via `pandas_ta`  
- 🌐 Simple web interface built with Streamlit

## 🧠 Machine Learning Models

- **Linear Regression**
- **Random Forest**
- *(Optional: LSTM via TensorFlow)*

The app compares predictions based on the selected model and plots future Bitcoin prices.

## 📁 Project Structure and Setup

```bash
bitcoin-price-prediction/
├── app.py                      # Main Streamlit app
├── ml_models.py               # ML model definitions
├── technical_indicators.py    # Technical indicators using pandas_ta
├── visualization.py           # Plotting/chart utilities
├── data_fetcher.py            # Fetches Bitcoin data via yfinance
├── utils.py                   # Helper functions
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignored files
└── README.md                  # Project documentation
```
# 1. Clone the Repository
git clone https://github.com/your-username/bitcoin-price-prediction.git
cd bitcoin-price-prediction

# 2. Create & Activate Virtual Environment
python -m venv venv
.\venv\Scripts\activate         # For Windows
# source venv/bin/activate      # For macOS/Linux

# 3. Install Required Libraries
pip install -r requirements.txt

# 4. Run the Streamlit App
streamlit run app.py

# Then open in your browser:
# http://localhost:8501
