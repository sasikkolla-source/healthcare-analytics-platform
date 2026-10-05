# Healthcare Analytics Platform

A Streamlit dashboard for hospital analytics with readmission risk prediction.
Uses synthetic patient data, so no dataset is needed.

## Features
- Overview KPIs, monthly admissions, department and age breakdowns
- Clinical insights by diagnosis and department
- Cost analysis with CSV download
- Random Forest readmission risk predictor

## Setup
```
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Using real data
Replace `load_data()` in `app.py` with `pd.read_csv("your_file.csv")` using these columns:
patient_id, admit_date, age, gender, department, diagnosis, length_of_stay, comorbidities, cost, readmitted

Demo only. Not for clinical decision-making.
