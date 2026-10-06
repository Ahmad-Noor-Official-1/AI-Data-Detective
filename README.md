# 🕵️ AI Data Detective

AI Data Detective is a Streamlit web app that turns CSV and Excel datasets into a clear data investigation. It checks data quality, detects anomalies, analyzes trends, creates visual analytics, generates findings and recommendations, and produces a downloadable PDF investigation report.

## Features

- 📂 CSV and Excel upload
- 🔎 Dataset preview and structure analysis
- 🧹 Missing-value and duplicate detection
- 🚨 Anomaly detection with Isolation Forest
- 📈 Trend Detective for time-based numerical data
- 🧠 Smart Detective Findings
- 📊 Bar, line, pie, and scatter visualizations
- 📄 Professional PDF Investigation Report
- 💡 Data-driven recommendations
- 🌙 Professional dark SaaS-style interface

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL
- ReportLab

## Run Locally

1. Clone this repository.
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

## Deployment

This project can be deployed for free using Streamlit Community Cloud. Connect the GitHub repository, select `app.py` as the main file, and deploy.

## Project Purpose

AI Data Detective was built as a practical data/AI project demonstrating Python development, data analysis, machine learning, visualization, and report automation in one usable application.
