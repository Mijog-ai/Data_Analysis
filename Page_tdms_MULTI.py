import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from nptdms import TdmsFile
import numpy as np
import csv
import base64


def load_and_process_tdms_file(file):
    with TdmsFile.open(file) as tdms_file:
        groups = tdms_file.groups()
        data_dict = {}
        for group in groups:
            for channel in group.channels():
                channel_name = f"{group.name}/{channel.name}"
                data = channel[:]
                data_dict[channel_name] = data

    max_length = max(len(data) for data in data_dict.values())
    for key in data_dict:
        if len(data_dict[key]) < max_length:
            pad_length = max_length - len(data_dict[key])
            data_dict[key] = np.pad(data_dict[key], (0, pad_length), 'constant', constant_values=np.nan)

    df = pd.DataFrame(data_dict)

    st.write("DataFrame shape:", df.shape)
    st.write("DataFrame columns:", df.columns.tolist())
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    return df


def plot_data(df, x_column, y_columns, title="Data Visualization"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in y_columns:
        ax.plot(df[x_column], df[column], label=column)
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    return fig


def get_image_download_link(fig, filename, text):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'


def dataframe_to_csv(df):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(df.columns)
    for index, row in df.iterrows():
        writer.writerow(row.values)
    return output.getvalue()


def main():
    st.title("TDMS File Analyzer with Enhanced Plotting")

    uploaded_file = st.file_uploader("Choose a .tdms file", type="tdms")

    if uploaded_file is not None:
        try:
            df = load_and_process_tdms_file(uploaded_file)

            if df is not None and not df.empty:
                st.write("Data Preview:")
                st.dataframe(df.head())

                st.write("Data Statistics:")
                st.dataframe(df.describe())

                st.write("Full Data:")
                st.dataframe(df)

                st.write("Dynamic Data Visualization:")
                columns = df.columns.tolist()

                x_column = st.selectbox("Select X-axis (usually time)", options=columns, index=0)
                y_columns = st.multiselect("Select Y-axis column(s)", options=columns,
                                           default=[columns[1]] if len(columns) > 1 else [])

                if x_column and y_columns:
                    fig = plot_data(df, x_column, y_columns)
                    st.pyplot(fig)

                    # Option to save the plot
                    st.markdown(get_image_download_link(fig, "plot.png", "Download Plot"), unsafe_allow_html=True)

                # Generate CSV using custom function
                csv_string = dataframe_to_csv(df)

                st.download_button(
                    label="Download data as CSV",
                    data=csv_string,
                    file_name="processed_data.csv",
                    mime="text/csv",
                )

                st.write("Preview of CSV content:")
                st.text(csv_string[:1000])

                st.write("CSV string length:", len(csv_string))
                st.write("Number of newlines in CSV:", csv_string.count('\n'))
                st.write("Number of commas in first line of CSV:", csv_string.split('\n')[0].count(','))

            else:
                st.error("Failed to process the file. Please check the debug information above.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Please check the file format and try again.")


if __name__ == "__main__":
    main()