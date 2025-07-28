import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set page config
st.set_page_config(page_title="ACARS Spoofed Packet Detector", page_icon="✈️", layout="wide")
st.title("✈️ ACARS Spoofed Packet Detection Dashboard")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload your ACARS packet CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

    # --- Step 2: Data Preprocessing & Basic Info ---
    st.header("Data Preview")
    st.dataframe(df.head())

    st.header("Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.header("Missing Values")
    st.write(df.isnull().sum())

    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # --- Step 3: Exploratory Data Analysis ---
    st.header("Exploratory Data Analysis")

    st.subheader("Spoofed vs Normal Packets")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='IsSpoofed', ax=ax)
    ax.set_xticklabels(['Normal', 'Spoofed'])
    ax.set_xlabel("Packet Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Spoofed vs Normal Packets")
    st.pyplot(fig)

    st.subheader("Altitude Distribution by Spoofing Status")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x='Altitude_ft', hue='IsSpoofed', bins=50, kde=True, ax=ax)
    ax.set_xlabel("Altitude (feet)")
    st.pyplot(fig)

    st.subheader("Speed Distribution by Spoofing Status")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x='Speed_knots', hue='IsSpoofed', bins=50, kde=True, ax=ax)
    ax.set_xlabel("Speed (knots)")
    st.pyplot(fig)

    st.subheader("Geospatial Distribution of Packets")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='IsSpoofed', palette='coolwarm', alpha=0.6, ax=ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

    st.subheader("Spoofing Rate by Message Type")
    spoof_by_type = df.groupby('MessageType')['IsSpoofed'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    spoof_by_type.plot(kind='bar', color='orange', ax=ax)
    ax.set_ylabel("Spoofing Rate")
    ax.set_xlabel("Message Type")
    ax.set_title("Spoofing Rate by Message Type")
    st.pyplot(fig)

    st.subheader("Spoofed Packets Over Time")
    df_ts = df.copy()
    df_ts.set_index('Timestamp', inplace=True)
    spoofed_ts = df_ts[df_ts['IsSpoofed'] == 1].resample('10T').size()
    fig, ax = plt.subplots(figsize=(10, 4))
    spoofed_ts.plot(ax=ax)
    ax.set_ylabel("Number of Spoofed Packets")
    ax.set_xlabel("Time")
    st.pyplot(fig)

    # --- Step 4: Prepare data for ML model ---
    st.header("Machine Learning Model to Detect Spoofed Packets")

    df_ml = df.copy()

    # Add 'Unknown' to each categorical field
    for col in ['MessageType', 'FlightNumber', 'AircraftID']:
        df_ml[col] = df_ml[col].astype('category')
        df_ml[col] = df_ml[col].cat.add_categories('Unknown')
        df_ml[col] = df_ml[col].fillna('Unknown')

    # Encode categories
    df_ml['MessageType'] = df_ml['MessageType'].astype('category').cat.codes
    df_ml['FlightNumber'] = df_ml['FlightNumber'].astype('category').cat.codes
    df_ml['AircraftID'] = df_ml['AircraftID'].astype('category').cat.codes

    # Drop rows with missing numerical values
    df_ml = df_ml.dropna(subset=['Latitude', 'Longitude', 'Altitude_ft', 'Speed_knots', 'IsSpoofed'])

    features = ['Latitude', 'Longitude', 'Altitude_ft', 'Speed_knots', 'MessageType', 'FlightNumber', 'AircraftID']
    X = df_ml[features]
    y = df_ml['IsSpoofed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {acc:.3f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # --- Step 5: Predict new packet spoofing ---
    st.header("Test New Packet(s)")
    tab1, tab2 = st.tabs(["📥 Upload CSV of Packets", "✍️ Enter Manually"])

    with tab1:
        st.subheader("Batch Prediction from CSV")
        uploaded_test_file = st.file_uploader("Upload a CSV file with new packets for prediction", type=["csv"], key="test_csv")

        if uploaded_test_file:
            test_df = pd.read_csv(uploaded_test_file)

            try:
                # Handle unknowns
                for col in ['MessageType', 'FlightNumber', 'AircraftID']:
                    known_cats = df[col].astype('category').cat.categories.tolist() + ['Unknown']
                    test_df[col] = test_df[col].apply(lambda x: x if x in known_cats else 'Unknown')
                    test_df[col] = pd.Categorical(test_df[col], categories=known_cats).codes

                batch_features = ['Latitude', 'Longitude', 'Altitude_ft', 'Speed_knots', 'MessageType', 'FlightNumber', 'AircraftID']
                predictions = clf.predict(test_df[batch_features])
                probabilities = clf.predict_proba(test_df[batch_features])[:, 1]

                test_df['Predicted_IsSpoofed'] = predictions
                test_df['Spoof_Probability'] = probabilities

                st.subheader("Prediction Results")
                st.dataframe(test_df)

                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="spoofed_predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error processing the file: {e}")

    with tab2:
        st.subheader("Manual Packet Entry")
        with st.form("manual_entry_form"):
            input_lat = st.number_input("Latitude (-90 to 90)", min_value=-90.0, max_value=90.0, format="%.4f")
            input_lon = st.number_input("Longitude (-180 to 180)", min_value=-180.0, max_value=180.0, format="%.4f")
            input_alt = st.number_input("Altitude (feet)", min_value=-100000, max_value=100000, step=1)
            input_speed = st.number_input("Speed (knots)", min_value=-1000, max_value=10000, step=1)

            input_msgtype = st.text_input("Message Type (e.g., ATC, WX, POS, etc.)")
            input_flightnum = st.text_input("Flight Number (e.g., BA2490, DL112)")
            input_aircraftid = st.text_input("Aircraft ID (e.g., N12345)")

            submitted = st.form_submit_button("Predict Spoofing")

            if submitted:
                # Same logic used in training for encoding
                msgtype_cats = list(df['MessageType'].astype('category').cat.categories) + ['Unknown']
                flightnum_cats = list(df['FlightNumber'].astype('category').cat.categories) + ['Unknown']
                aircraftid_cats = list(df['AircraftID'].astype('category').cat.categories) + ['Unknown']

                msgtype_code = msgtype_cats.index(input_msgtype) if input_msgtype in msgtype_cats else msgtype_cats.index('Unknown')
                flightnum_code = flightnum_cats.index(input_flightnum) if input_flightnum in flightnum_cats else flightnum_cats.index('Unknown')
                aircraftid_code = aircraftid_cats.index(input_aircraftid) if input_aircraftid in aircraftid_cats else aircraftid_cats.index('Unknown')

                new_data = pd.DataFrame({
                    'Latitude': [input_lat],
                    'Longitude': [input_lon],
                    'Altitude_ft': [input_alt],
                    'Speed_knots': [input_speed],
                    'MessageType': [msgtype_code],
                    'FlightNumber': [flightnum_code],
                    'AircraftID': [aircraftid_code],
                })

                prediction = clf.predict(new_data)[0]
                proba = clf.predict_proba(new_data)[0][1]

                if prediction == 1:
                    st.error(f"🚨 This packet is likely SPOOFED with probability {proba:.2%}")
                else:
                    st.success(f"✅ This packet is likely NORMAL with probability {(1 - proba):.2%}")

else:
    st.info("Please upload the ACARS packets CSV file to get started.")
