import streamlit as st
import pandas as pd
import os
import sys

# Add the parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_kaggle_dataset, load_dataset_from_path, load_dataset_from_upload, get_data_info
from utils.visualizer import plot_specialty_distribution, plot_text_length_distribution

# Page configuration
st.set_page_config(
    page_title="Data Exploration - Medical Transcription NLP",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None

st.title("Data Exploration")
st.write("Load and explore the medical transcription dataset.")

# Data loading options
st.header("Load Data")

loading_method = st.radio(
    "Select data loading method:",
    ["Load from Kaggle", "Upload CSV file"]
)

if loading_method == "Load from Kaggle":
    if st.button("Load Medical Transcription Data from Kaggle"):
        dataset_path = load_kaggle_dataset()
        if dataset_path:
            st.success(f"Dataset downloaded successfully to: {dataset_path}")
            df = load_dataset_from_path(dataset_path)
            if df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_info = get_data_info(df)
                st.success(f"Dataset loaded successfully with {df.shape[0]} records.")
                st.rerun()

elif loading_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_dataset_from_upload(uploaded_file)
        if df is not None:
            st.session_state.dataset = df
            st.session_state.dataset_info = get_data_info(df)
            st.success(f"Dataset loaded successfully with {df.shape[0]} records.")
            st.rerun()

# Display dataset information
if st.session_state.dataset is not None:
    df = st.session_state.dataset
    info = st.session_state.dataset_info
    
    st.header("Dataset Overview")
    
    # Basic dataset information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {info['shape'][0]}, Columns: {info['shape'][1]}")
    
    with col2:
        st.subheader("Columns")
        st.write(info['columns'])
    
    # Dataset preview
    st.subheader("Data Preview")
    st.dataframe(info['sample'])
    
    # Missing values
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Column': list(info['missing_values'].keys()),
        'Missing Values': list(info['missing_values'].values())
    })
    st.dataframe(missing_df)
    
    # Data visualization
    st.header("Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'medical_specialty' in df.columns:
            st.subheader("Medical Specialty Distribution")
            fig = plot_specialty_distribution(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to create medical specialty distribution plot.")
    
    with col2:
        # Determine appropriate text column
        text_columns = [col for col in df.columns if df[col].dtype == 'object' and col != 'medical_specialty']
        if text_columns:
            selected_column = st.selectbox("Select text column for length analysis:", text_columns)
            
            st.subheader(f"{selected_column} Length Distribution")
            fig = plot_text_length_distribution(df, text_column=selected_column)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Unable to create {selected_column} length distribution plot.")
    
    # Data summary
    if 'medical_specialty' in df.columns:
        st.subheader("Medical Specialty Summary")
        specialty_counts = df['medical_specialty'].value_counts().reset_index()
        specialty_counts.columns = ['Specialty', 'Count']
        st.dataframe(specialty_counts)
    
    # Column statistics
    st.subheader("Column Data Types")
    dtypes_df = pd.DataFrame({
        'Column': list(info['dtypes'].keys()),
        'Data Type': [str(dtype) for dtype in info['dtypes'].values()]
    })
    st.dataframe(dtypes_df)

else:
    st.info("Please load a dataset to explore.")
