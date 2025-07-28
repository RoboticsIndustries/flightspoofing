# ACARS Spoofed Packet Detection Dashboard

This Streamlit app allows you to upload ACARS packet data and analyze it to detect spoofed packets using machine learning.

## Features

- Upload a CSV file containing ACARS packets data (including spoofed labels).
- Visualize exploratory data analysis:
  - Spoofed vs Normal packet counts
  - Altitude and speed distributions by spoofing status
  - Geospatial scatter plot
  - Spoofing rates by message type
  - Time series of spoofed packets
- Train a Random Forest classifier on the uploaded data to detect spoofed packets.
- Input new packet data to predict if it is spoofed or not in real time.

## How to run locally

1. Clone this repository or download the `streamlit_app.py` file.

2. Install dependencies (recommended to use a virtual environment):

```bash
pip install streamlit pandas matplotlib seaborn scikit-learn
