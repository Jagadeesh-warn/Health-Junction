import streamlit as st
import os
import sys

# Add the current directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_sample_data

# Page configuration
st.set_page_config(
    page_title="Medical Transcription NLP Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Medical Transcription NLP Analysis")

st.markdown("""
## Welcome to the Medical Transcription NLP Analysis App

This application provides tools for analyzing medical transcription data using Natural Language Processing techniques.
The application is organized into several sections accessible from the sidebar:

- **Data Exploration**: Load and explore the medical transcription dataset
- **Text Analysis**: Perform basic text analysis on medical transcriptions
- **NLP Pipeline**: Process text through NLP pipelines for detailed analysis
- **Named Entity Recognition**: Identify medical entities in the transcriptions
- **Topic Modeling**: Discover topics in the medical transcription corpus
- **Medical Visualizations**: View interactive visualizations of medical data
- **Advanced Visualizations**: Explore advanced NLP visualizations

### Getting Started
1. Use the sidebar to navigate between different analysis sections
2. In the Data Exploration page, load your data or use sample data
3. Explore the different analysis options in each section
""")

# Display sample data for quick start
st.subheader("Sample Data Preview")
try:
    sample_df = load_sample_data()
    if sample_df is not None and not sample_df.empty:
        st.dataframe(sample_df.head(5))
    else:
        st.warning("No sample data available. Please go to the Data Exploration page to load data.")
except Exception as e:
    st.error(f"Error loading sample data: {e}")
    st.info("Please navigate to the Data Exploration page to load your data.")

# Sidebar with app info
with st.sidebar:
    st.title("Navigation")
    st.info("Use the pages in the sidebar to navigate to different sections of the application.")
    
    st.markdown("### About")
    st.markdown("""
    This application provides NLP analysis tools for medical transcriptions.
    
    Data source: [Medical Transcriptions on Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
    """)

    st.markdown("### Technologies Used")
    st.markdown("""
    - Streamlit
    - SpaCy
    - NLTK
    - Pandas
    - Matplotlib/Plotly
    """)
