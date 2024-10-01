import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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


def plot_data(df, x_column, y_column):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x_column], df[y_column])
    ax.set_title(f'{y_column} vs {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    plt.tight_layout()
    return fig


def main():
    st.title("MZ .asc File Loader and Dynamic Analyzer")

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

                if x_column and y_column:
                    fig = plot_data(df, x_column, y_column)
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