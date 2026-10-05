"""
Healthcare Analytics Platform (simple base)
-------------------------------------------
Run:
    pip install streamlit pandas numpy plotly scikit-learn
    streamlit run healthcare_analytics_platform.py

Uses SYNTHETIC data. To use real data, replace load_data() with
pd.read_csv("your_file.csv") keeping the same column names.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Healthcare Analytics Platform", layout="wide")

DEPARTMENTS = ["Cardiology", "Orthopedics", "Neurology", "Oncology", "Pediatrics", "General Medicine"]
DIAGNOSES = ["Hypertension", "Diabetes", "Fracture", "Stroke", "Cancer", "Asthma", "Infection"]


# ---------------------------------------------------------------- DATA
@st.cache_data
def load_data(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    admit = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D")
    age = np.clip(rng.normal(50, 20, n), 1, 95).astype(int)
    los = np.clip(rng.gamma(2.0, 2.5, n) + age / 30, 1, 30).round().astype(int)
    comorbidities = rng.poisson(1.2, n)
    cost = (los * rng.normal(900, 150, n) + comorbidities * 400).round(2)

    risk = 0.02 * (age - 50) + 0.25 * comorbidities + 0.08 * los - 1.8
    readmitted = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)

    return pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "admit_date": admit,
        "age": age,
        "gender": rng.choice(["Male", "Female"], n),
        "department": rng.choice(DEPARTMENTS, n),
        "diagnosis": rng.choice(DIAGNOSES, n),
        "length_of_stay": los,
        "comorbidities": comorbidities,
        "cost": cost,
        "readmitted": readmitted,
    })


# ---------------------------------------------------------------- MODEL
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = pd.get_dummies(df[["age", "length_of_stay", "comorbidities", "gender", "department"]], drop_first=True)
    y = df["readmitted"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
    model = RandomForestClassifier(n_estimators=150, random_state=1).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, list(X.columns), auc


# ---------------------------------------------------------------- UI
df = load_data()

st.title("🏥 Healthcare Analytics Platform")
st.caption("Hospital operations dashboard with readmission risk prediction (synthetic data)")

# Sidebar filters
st.sidebar.header("Filters")
depts = st.sidebar.multiselect("Department", DEPARTMENTS, default=DEPARTMENTS)
genders = st.sidebar.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
age_min, age_max = st.sidebar.slider("Age range", 0, 100, (0, 100))
dates = st.sidebar.date_input("Admission date", (df.admit_date.min(), df.admit_date.max()))

f = df[df.department.isin(depts) & df.gender.isin(genders) & df.age.between(age_min, age_max)]
if isinstance(dates, tuple) and len(dates) == 2:
    f = f[(f.admit_date >= pd.to_datetime(dates[0])) & (f.admit_date <= pd.to_datetime(dates[1]))]

if f.empty:
    st.warning("No records match the selected filters.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧬 Clinical Insights", "💰 Cost Analysis", "🤖 Readmission Risk"])

# --- Overview
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{len(f):,}")
    c2.metric("Avg Length of Stay", f"{f.length_of_stay.mean():.1f} days")
    c3.metric("Readmission Rate", f"{f.readmitted.mean():.1%}")
    c4.metric("Avg Cost / Patient", f"₹{f.cost.mean():,.0f}")

    monthly = f.set_index("admit_date").resample("MS").size().reset_index(name="admissions")
    st.plotly_chart(px.line(monthly, x="admit_date", y="admissions", markers=True,
                            title="Monthly Admissions"), use_container_width=True)

    col_a, col_b = st.columns(2)
    col_a.plotly_chart(px.pie(f, names="department", title="Patients by Department", hole=0.4),
                       use_container_width=True)
    col_b.plotly_chart(px.histogram(f, x="age", nbins=20, color="gender", title="Age Distribution"),
                       use_container_width=True)

# --- Clinical Insights
with tab2:
    diag = f.groupby("diagnosis").agg(patients=("patient_id", "count"),
                                      avg_los=("length_of_stay", "mean"),
                                      readmit_rate=("readmitted", "mean")).reset_index()
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(px.bar(diag, x="diagnosis", y="patients", title="Cases by Diagnosis"),
                       use_container_width=True)
    col_b.plotly_chart(px.bar(diag, x="diagnosis", y="readmit_rate", title="Readmission Rate by Diagnosis"),
                       use_container_width=True)
    st.plotly_chart(px.box(f, x="department", y="length_of_stay", title="Length of Stay by Department"),
                    use_container_width=True)

# --- Cost Analysis
with tab3:
    cost_dept = f.groupby("department")["cost"].agg(["sum", "mean"]).reset_index()
    col_a, col_b = st.columns(2)
    col_a.plotly_chart(px.bar(cost_dept, x="department", y="sum", title="Total Cost by Department"),
                       use_container_width=True)
    col_b.plotly_chart(px.scatter(f.sample(min(800, len(f))), x="length_of_stay", y="cost",
                                  color="department", title="Cost vs Length of Stay"),
                       use_container_width=True)
    st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False), "filtered_patients.csv")

# --- Prediction
with tab4:
    model, cols, auc = train_model(df)
    st.write(f"Random Forest model · ROC-AUC on held-out data: **{auc:.2f}**")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 1, 100, 60)
    los = c2.number_input("Length of stay (days)", 1, 30, 5)
    com = c3.number_input("Comorbidities", 0, 10, 2)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    dept = c2.selectbox("Department", DEPARTMENTS)

    if st.button("Predict readmission risk"):
        row = pd.DataFrame([{"age": age, "length_of_stay": los, "comorbidities": com,
                             "gender": gender, "department": dept}])
        row = pd.get_dummies(row).reindex(columns=cols, fill_value=0)
        p = model.predict_proba(row)[0, 1]
        level = "High" if p > 0.6 else "Medium" if p > 0.35 else "Low"
        st.metric("Readmission probability", f"{p:.1%}", level + " risk")
        st.progress(float(p))

    imp = pd.Series(model.feature_importances_, index=cols).nlargest(8).reset_index()
    imp.columns = ["feature", "importance"]
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                           title="Top Risk Factors"), use_container_width=True)

st.sidebar.info("⚠️ Demo only. Not for clinical decision-making.")
