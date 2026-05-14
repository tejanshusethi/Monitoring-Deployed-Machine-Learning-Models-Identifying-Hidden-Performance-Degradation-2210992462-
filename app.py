import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ML Monitoring System", layout="wide")

st.title("🚀 Real-Time Model Monitoring System")
st.write("Dynamic Drift Detection & Performance Monitoring Dashboard")

# -----------------------------
# SESSION STATE (STORE OLD DATA)
# -----------------------------
if "prev_data" not in st.session_state:
    st.session_state.prev_data = None

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# DATA PREP
# -----------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL TRAIN
# -----------------------------
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# -----------------------------
# PERFORMANCE BEFORE
# -----------------------------
y_pred = model.predict(X_test)

acc_before = accuracy_score(y_test, y_pred)
prec_before = precision_score(y_test, y_pred)
rec_before = recall_score(y_test, y_pred)

# -----------------------------
# DRIFT SIMULATION
# -----------------------------
drifted_data = X_test + np.random.normal(0, 1, X_test.shape)

# -----------------------------
# PERFORMANCE AFTER
# -----------------------------
y_pred_drift = model.predict(drifted_data)

acc_after = accuracy_score(y_test, y_pred_drift)
prec_after = precision_score(y_test, y_pred_drift)
rec_after = recall_score(y_test, y_pred_drift)

# -----------------------------
# PERFORMANCE UI
# -----------------------------
st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Before Drift")
    st.metric("Accuracy", f"{acc_before:.4f}")
    st.metric("Precision", f"{prec_before:.4f}")
    st.metric("Recall", f"{rec_before:.4f}")

with col2:
    st.markdown("### After Drift")
    st.metric("Accuracy", f"{acc_after:.4f}")
    st.metric("Precision", f"{prec_after:.4f}")
    st.metric("Recall", f"{rec_after:.4f}")

if acc_after < acc_before:
    st.error("🚨 Performance dropped due to drift!")

st.markdown("---")

# -----------------------------
# PERFORMANCE GRAPH
# -----------------------------
st.subheader("📊 Performance Comparison")

labels = ["Accuracy", "Precision", "Recall"]
before = [acc_before, prec_before, rec_before]
after = [acc_after, prec_after, rec_after]

x = np.arange(len(labels))

fig, ax = plt.subplots()
ax.bar(x - 0.2, before, 0.4, label="Before")
ax.bar(x + 0.2, after, 0.4, label="After")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

st.pyplot(fig)

st.markdown("---")

# -----------------------------
# DYNAMIC DATA GENERATION
# -----------------------------
tools = ["Evidently AI", "WhyLabs", "SageMaker", "Fiddler AI"]

current_data = {
    "data_drift": np.random.randint(10, 30, 4),
    "concept_drift": np.random.randint(15, 35, 4),
    "cpu": np.random.randint(15, 45, 4),
    "memory": np.random.randint(250, 700, 4)
}

x = np.arange(len(tools))

# -----------------------------
# FIGURE 1 (LATENCY COMPARISON)
# -----------------------------
st.subheader("📊 Detection Latency (Dynamic Comparison)")

fig1, ax1 = plt.subplots()

ax1.bar(x - 0.2, current_data["data_drift"], 0.4, label="Data Drift (Now)")
ax1.bar(x + 0.2, current_data["concept_drift"], 0.4, label="Concept Drift (Now)")

if st.session_state.prev_data is not None:
    prev = st.session_state.prev_data
    ax1.plot(x, prev["data_drift"], linestyle='--', marker='o', label="Prev Data Drift")
    ax1.plot(x, prev["concept_drift"], linestyle='--', marker='o', label="Prev Concept Drift")

ax1.set_xticks(x)
ax1.set_xticklabels(tools)
ax1.legend()

st.pyplot(fig1)

# -----------------------------
# FIGURE 2 (CPU + MEMORY)
# -----------------------------
st.subheader("📊 System Usage (Dynamic Comparison)")

fig2, ax2 = plt.subplots()

ax2.bar(x, current_data["cpu"], label="CPU (Now)")

ax3 = ax2.twinx()
ax3.plot(x, current_data["memory"], marker='o', label="Memory (Now)")

if st.session_state.prev_data is not None:
    prev = st.session_state.prev_data
    ax2.plot(x, prev["cpu"], linestyle='--', marker='x', label="Prev CPU")
    ax3.plot(x, prev["memory"], linestyle='--', marker='x', label="Prev Memory")

ax2.set_xticks(x)
ax2.set_xticklabels(tools)

ax2.legend(loc="upper left")
ax3.legend(loc="upper right")

st.pyplot(fig2)

st.markdown("---")

# -----------------------------
# DYNAMIC TABLES
# -----------------------------
st.subheader("📋 Monitoring Tables (Dynamic)")

df1 = pd.DataFrame({
    "Tool": tools,
    "Data Drift (sec)": current_data["data_drift"],
    "Concept Drift (sec)": current_data["concept_drift"],
    "CPU (%)": current_data["cpu"],
    "Memory (MB)": current_data["memory"]
})

st.dataframe(df1, use_container_width=True)

false_positive = np.round(np.random.uniform(3, 8, 4), 2)
false_negative = np.round(np.random.uniform(3, 7, 4), 2)

df2 = pd.DataFrame({
    "Tool": tools,
    "False Positive (%)": false_positive,
    "False Negative (%)": false_negative
})

st.dataframe(df2, use_container_width=True)

st.info("Values are dynamically generated to simulate real-world changes.")

st.markdown("---")

# -----------------------------
# DRIFT DETECTION
# -----------------------------
st.subheader("🔍 Drift Detection")

changed = []

for i in range(X_train.shape[1]):
    if abs(X_train[:, i].mean() - drifted_data[:, i].mean()) > 0.5:
        changed.append(f"Feature {i}")

if changed:
    st.warning(f"Drift detected in: {changed}")
else:
    st.success("No major drift detected")

st.markdown("---")

# -----------------------------
# REPORT
# -----------------------------
st.subheader("📄 Generate Drift Report")

if st.button("Generate Report"):
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=pd.DataFrame(X_train),
        current_data=pd.DataFrame(drifted_data)
    )
    report.save_html("drift_report.html")
    st.success("Report generated!")

# -----------------------------
# RETRAIN
# -----------------------------
st.subheader("🔄 Retrain Model")

if st.button("Retrain Model"):
    model.fit(drifted_data, y_test)
    st.success("Model retrained!")

# -----------------------------
# SAVE CURRENT → PREVIOUS
# -----------------------------
st.session_state.prev_data = current_data

# -----------------------------
# REFRESH BUTTON
# -----------------------------
if st.button("🔄 Refresh Simulation"):
    st.rerun()