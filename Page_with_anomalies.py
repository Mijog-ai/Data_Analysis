import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import io


def load_and_process_asc_file(file):
    content = file.getvalue().decode("utf-8")
    lines = content.split('\n')

    # Find the start of the data
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("Messzeit[s]"):
            data_start = i + 1
            break

    # Extract header and data
    header = lines[data_start - 1].split('\t')
    data = [line.split('\t') for line in lines[data_start:] if line.strip()]

    # Rename duplicate columns
    new_header = []
    seen = {}
    for i, item in enumerate(header):
        if item in seen:
            seen[item] += 1
            new_header.append(f"{item}_{seen[item]}")
        else:
            seen[item] = 0
            new_header.append(item)

    # Create DataFrame with renamed columns
    df = pd.DataFrame(data, columns=new_header)

    # Debug information
    st.write("Original columns:", header)
    st.write("Renamed columns:", new_header)
    st.write("DataFrame shape:", df.shape)
    st.write("DataFrame info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Convert columns to appropriate types
    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            st.write(f"Error processing column {col}: {str(e)}")
            st.write(f"Sample data for {col}:", df[col].head())

    return df


def plot_data(df, x_column, y_column, anomalies=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x_column], df[y_column], label='Data')

    if anomalies is not None:
        ax.scatter(df[x_column][anomalies], df[y_column][anomalies], color='red', label='Anomalies', zorder=5)

    ax.set_title(f'{y_column} vs {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend()
    plt.tight_layout()
    return fig


def z_score_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return np.abs(z_scores) > threshold


def iqr_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (df[column] < lower_bound) | (df[column] > upper_bound)


def isolation_forest_outliers(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns].fillna(0))

    model = IsolationForest(contamination=0.01, random_state=42)
    preds = model.fit_predict(df_scaled)

    return preds == -1


def dbscan_outliers(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns].fillna(0))

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    preds = dbscan.fit_predict(df_scaled)

    return preds == -1  # Noise points are marked as -1


def local_outlier_factor(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns].fillna(0))

    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    preds = lof.fit_predict(df_scaled)

    return preds == -1  # Anomalies are marked as -1


def one_class_svm(df, columns):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns].fillna(0))

    model = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    preds = model.fit_predict(df_scaled)

    return preds == -1  # Anomalies are marked as -1


def main():
    st.title("MZ .asc File Loader and Dynamic Analyzer with Anomaly Detection")

    uploaded_file = st.file_uploader("Choose a .asc file", type="asc")

    if uploaded_file is not None:
        try:
            df = load_and_process_asc_file(uploaded_file)

            if df is not None and not df.empty:
                st.write("Data Preview:")
                st.dataframe(df.head())

                st.write("Data Statistics:")
                st.dataframe(df.describe())

                st.write("Dynamic Data Visualization:")
                columns = df.columns.tolist()

                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("Select X-axis", options=columns,
                                            index=columns.index("Messzeit[s]") if "Messzeit[s]" in columns else 0)
                with col2:
                    y_column = st.selectbox("Select Y-axis", options=columns, index=1 if len(columns) > 1 else 0)

                # Anomaly detection options
                st.write("Anomaly Detection Methods:")
                anomaly_method = st.selectbox(
                    "Select anomaly detection method",
                    options=["None", "Z-score", "IQR", "Isolation Forest", "DBSCAN", "Local Outlier Factor", "One-Class SVM"]
                )

                anomalies = None
                if anomaly_method != "None":
                    if anomaly_method == "Z-score":
                        threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0)
                        anomalies = z_score_outliers(df, y_column, threshold)
                    elif anomaly_method == "IQR":
                        anomalies = iqr_outliers(df, y_column)
                    elif anomaly_method == "Isolation Forest":
                        anomalies = isolation_forest_outliers(df, [x_column, y_column])
                    elif anomaly_method == "DBSCAN":
                        anomalies = dbscan_outliers(df, [x_column, y_column])
                    elif anomaly_method == "Local Outlier Factor":
                        anomalies = local_outlier_factor(df, [x_column, y_column])
                    elif anomaly_method == "One-Class SVM":
                        anomalies = one_class_svm(df, [x_column, y_column])

                    st.write(f"Number of anomalies detected: {np.sum(anomalies)}")

                # Plot data with anomalies
                fig = plot_data(df, x_column, y_column, anomalies=anomalies)
                st.pyplot(fig)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to process the file. Please check the debug information above.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Please check the file format and try again.")


if __name__ == "__main__":
    main()
