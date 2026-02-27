⚡ AI-Based Energy Consumption Forecasting Dashboard

A complete machine learning pipeline and interactive Streamlit dashboard for forecasting electricity consumption using historical smart meter data.


---

🚀 Project Overview

Electricity demand changes due to seasonal trends, daily behavior patterns, and long-term usage variations.

This project builds an end-to-end forecasting system that:

Cleans and processes 2M+ smart meter records

Performs time-series analysis

Compares baseline and machine learning models

Predicts future electricity consumption

Provides an interactive web dashboard

Allows CSV download of forecast results



---

📊 Dataset

Individual Household Electric Power Consumption Dataset (2006–2010)

⚠️ The dataset is not included in this repository due to GitHub file size limits.

Download it from:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

After downloading, place the file inside:

data/household_power_consumption.txt


---

🧠 Methodology

1️⃣ Data Processing

Missing value handling

Type conversion

Time indexing

Daily resampling


2️⃣ Exploratory Data Analysis

Seasonal trend detection

Peak demand identification

Weekday vs weekend comparison


3️⃣ Forecasting Models

Baseline Moving Average

Linear Regression (time-based feature)



---

📉 Model Performance

Baseline MAE: ~0.56
Linear Regression MAE: ~0.24

Linear Regression reduced prediction error by more than 50% compared to the baseline model.


---

🖥 Dashboard Features

Daily Energy Trend Visualization

Model Performance Metrics

Forecast vs Actual Comparison

Future Power Consumption Prediction

Peak Demand Detection

Download Forecast Results as CSV



---

🛠 Tech Stack

Python
Pandas
NumPy
Matplotlib
Scikit-learn
Streamlit


---

📂 Project Structure

AI_Energy_Forecasting/

analysis.py  → Data loading and EDA

models.py    → Model training and evaluation

main.py      → Script execution version

app.py       → Streamlit dashboard

requirements.txt

README.md



---

▶ How to Run Locally

1. Clone the repository: git clone <your-repo-link>


2. Install dependencies: pip install -r requirements.txt


3. Run the dashboard: streamlit run app.py




---

🔮 Future Improvements

Add seasonal features (Month, Weekday, Year)

Implement ARIMA or Prophet model

Improve forecast accuracy

Deploy for public access



---

👨‍💻 Author

Mani
Electrical & Electronics Engineering

---

⭐ If you found this project useful, consider giving it a star..
