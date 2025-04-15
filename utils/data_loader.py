import pandas as pd
import streamlit as st
import os
import tempfile
import kagglehub
import time
import random

@st.cache_data
def load_sample_data():
    """
    Load a small sample of the medical transcriptions dataset for demonstration purposes.
    Uses cached data to avoid reloading on every rerun.
    
    Returns:
        pandas.DataFrame: Sample of medical transcriptions
    """
    try:
        # Create a small sample dataframe for initial viewing
        sample_data = {
            'description': [
                "Patient presents with chest pain and shortness of breath.",
                "Follow-up visit for diabetes management.",
                "MRI results show mild degenerative changes.",
                "Patient reports headaches and dizziness.",
                "Annual physical examination with no significant findings."
            ],
            'medical_specialty': [
                "Cardiovascular / Pulmonary",
                "Endocrinology",
                "Orthopedic",
                "Neurology",
                "General Medicine"
            ],
            'sample_name': [
                "Sample A",
                "Sample B",
                "Sample C",
                "Sample D",
                "Sample E"
            ]
        }
        
        return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error creating sample data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_kaggle_dataset():
    """
    Load the medical transcriptions dataset from Kaggle.
    Uses cached data to avoid reloading on every rerun.
    
    Returns:
        str: Path to the dataset files
    """
    try:
        with st.spinner("Downloading dataset from Kaggle..."):
            path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
        return path
    except Exception as e:
        st.error(f"Error downloading dataset from Kaggle: {e}")
        return None

def load_dataset_from_path(path):
    """
    Load the dataset from the specified path.
    
    Args:
        path (str): Path to the dataset files
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        if path and os.path.exists(path):
            # Get all CSV files in the path
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            if not csv_files:
                st.warning(f"No CSV files found in {path}")
                return None
            
            # Load the first CSV file
            file_path = os.path.join(path, csv_files[0])
            df = pd.read_csv(file_path)
            return df
        else:
            st.warning("Invalid path provided.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def load_dataset_from_upload(uploaded_file):
    """
    Load dataset from an uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        if uploaded_file is not None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load the dataset from the temporary file
            df = pd.read_csv(tmp_path)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
            
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None

def get_data_info(df):
    """
    Get information about the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset
        
    Returns:
        dict: Dataset information
    """
    if df is None or df.empty:
        return None
    
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample": df.head(5)
    }
    
    return info
