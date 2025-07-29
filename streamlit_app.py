import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Helper function to parse df.info
def get_df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer, memory_usage='deep')
    lines = buffer.getvalue().splitlines()
    header = lines[:3]
    footer = [lines[-1]]
    rows = []
    for line in lines[3:-1]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({'Column': parts[1], 'Non‑Null': parts[2], 'Dtype': parts[-1]})
    return header, pd.DataFrame(rows), footer

st.set_page_config(page_title="Telemetry Spoof Detection", page_icon="✈️", layout="wide")
st.title("✈️ Telemetry Spoof Detection Dashboard")

# Step 1: Upload telemetry CSV for multiple types
uploaded_file = st.file_uploader(
    "Upload telemetry CSV (ACARS, synthetic flight, or GPS spoofing data)", type=["csv"]
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded data: {len(df)} rows × {len(df.columns)} cols")

    data_type = st.selectbox("Select dataset type", ["ACARS/ADS‑B", "Synthetic Flight", "GPS Spoofing"])

    # Display data info
    st.header("📄 Data Info")
    header, info_df, footer = get_df_info(df)
    for line in header: st.text(line)
    st.dataframe(info_df, use_container_width=True)
    for line in footer: st.text(line)

    st.header("Missing Values")
    st.write(df.isnull().sum())

    # --- Feature selection by dataset type ---
    if data_type == "GPS Spoofing":
        features = []
        for c in ['Pseudorange','Carrier_Drift','CN0','Multipath','Lock_Time']:
            if c in df.columns: features.append(c)
        label = 'IsSpoofed'
    else:
        # common ACARS/ADS‑B & synthetic flight fields
        for col in ['MessageType','FlightNumber','AircraftID']:
            if col in df.columns:
                df[col] = df[col].astype('category').fillna('Unknown').cat.codes
        features = [f for f in ['Latitude','Longitude','Altitude_ft','Speed_knots','MessageType'] if f in df.columns]
        label = 'IsSpoofed'

    # Drop rows missing features or label
    df_ml = df.dropna(subset=features + [label])

    X = df_ml[features]
    y = df_ml[label]

    st.header("Model Training & Evaluation")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"**Accuracy:** {acc:.3f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Prediction interface
    st.header("Predict New Records")
    uploaded_test = st.file_uploader("Upload CSV for predictions", type=["csv"], key="test")
    if uploaded_test:
        test_df = pd.read_csv(uploaded_test)
        # preprocess categories if present
        if data_type != "GPS Spoofing":
            for col in ['MessageType','FlightNumber','AircraftID']:
                if col in test_df.columns:
                    cat = df[col].astype('category').cat.categories
                    test_df[col] = test_df[col].apply(lambda x: x if x in cat else 'Unknown')
                    test_df[col] = pd.Categorical(test_df[col], categories=cat).codes
        batch = test_df.dropna(subset=features)
        preds = clf.predict(batch[features])
        probs = clf.predict_proba(batch[features])[:,1]
        batch['Predicted_IsSpoofed'] = preds
        batch['Spoof_Prob'] = probs
        st.dataframe(batch)
        csv = batch.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions", data=csv, file_name="predictions.csv", mime="text/csv")

else:
    st.info("Upload a telemetry CSV file to get started.")
