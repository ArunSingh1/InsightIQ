import pandas as pd
import streamlit as st

def plot_csv_data(csv_path: str):
    """
    Dynamically plots data from a CSV file using Streamlit.
    Args:
        csv_path (str): Path to the CSV file to be plotted.
    """
    # Read the CSV file
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"
    
    # Streamlit App
    st.title("Dynamic Data Visualization")
    st.write(f"Data Preview ({csv_path}):")
    st.dataframe(data)

    # Column Selection
    st.sidebar.title("Visualization Settings")
    x_column = st.sidebar.selectbox("Select X-axis column:", data.columns)
    y_column = st.sidebar.selectbox("Select Y-axis column:", data.columns)
    chart_type = st.sidebar.selectbox("Select chart type:", ["Line Chart", "Bar Chart", "Scatter Plot"])

    # Plotting
    st.subheader("Generated Plot")
    if chart_type == "Line Chart":
        st.line_chart(data[[x_column, y_column]].set_index(x_column))
    elif chart_type == "Bar Chart":
        st.bar_chart(data[[x_column, y_column]].set_index(x_column))
    elif chart_type == "Scatter Plot":
        st.write(st.altair_chart(st.altair_chart(
            st.alt.Chart(data).mark_circle(size=60).encode(
                x=x_column,
                y=y_column,
                tooltip=list(data.columns)
            ).interactive()
        )))

    st.success("Visualization complete!")

# Example usage: Replace this with your desired CSV file path
# if __name__ == "__main__":
#     st.write("Dynamic CSV Visualization Tool")
#     csv_file_path = "data-1736495476779.csv"
#     plot_csv_data(csv_file_path)
