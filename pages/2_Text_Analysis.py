import streamlit as st
import pandas as pd
import os
import sys

# Add the parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processor import preprocess_text, get_word_frequency
from utils.visualizer import plot_word_frequency

# Page configuration
st.set_page_config(
    page_title="Text Analysis - Medical Transcription NLP",
    page_icon="📝",
    layout="wide"
)

st.title("Text Analysis")
st.write("Perform basic text analysis on medical transcriptions.")

# Check if dataset is loaded
if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Please go to the Data Exploration page to load your data.")
    st.stop()

df = st.session_state.dataset

# Determine appropriate text columns
text_columns = [col for col in df.columns if df[col].dtype == 'object']

if not text_columns:
    st.error("No text columns found in the dataset.")
    st.stop()

# Text analysis options
st.header("Text Analysis Options")

col1, col2 = st.columns(2)

with col1:
    selected_column = st.selectbox("Select text column for analysis:", text_columns)

with col2:
    analysis_type = st.selectbox(
        "Select analysis type:", 
        ["Word Frequency", "Sample Analysis", "Text Statistics"]
    )

# Sample selection
if 'sample_option' not in st.session_state:
    st.session_state.sample_option = "random"
if 'sample_index' not in st.session_state:
    st.session_state.sample_index = 0
if 'sample_text' not in st.session_state:
    st.session_state.sample_text = ""

st.header("Sample Selection")

sample_option = st.radio(
    "Select sample option:",
    ["Random Sample", "Enter Index", "Enter Custom Text"]
)

if sample_option == "Random Sample":
    st.session_state.sample_option = "random"
    if st.button("Get Random Sample"):
        if df.shape[0] > 0:
            st.session_state.sample_index = df.sample(1).index[0]
            st.session_state.sample_text = df.iloc[st.session_state.sample_index][selected_column]
            st.rerun()

elif sample_option == "Enter Index":
    st.session_state.sample_option = "index"
    max_index = df.shape[0] - 1
    sample_index = st.number_input("Enter sample index:", 0, max_index, st.session_state.sample_index)
    
    if sample_index != st.session_state.sample_index:
        st.session_state.sample_index = sample_index
        st.session_state.sample_text = df.iloc[sample_index][selected_column]
        st.rerun()

elif sample_option == "Enter Custom Text":
    st.session_state.sample_option = "custom"
    st.session_state.sample_text = st.text_area("Enter text for analysis:", st.session_state.sample_text, height=200)

# Display sample text
st.subheader("Sample Text")
if st.session_state.sample_text:
    st.write(st.session_state.sample_text)
else:
    st.info("No sample text selected.")

# Text preprocessing options
st.header("Text Preprocessing")

col1, col2 = st.columns(2)

with col1:
    remove_stopwords = st.checkbox("Remove stopwords", value=True)

with col2:
    lemmatize = st.checkbox("Lemmatize text", value=True)

# Perform text analysis
if st.session_state.sample_text:
    st.header("Analysis Results")
    
    # Preprocess text
    preprocessed_text = preprocess_text(
        st.session_state.sample_text, 
        remove_stopwords=remove_stopwords, 
        lemmatize=lemmatize
    )
    
    # Word frequency analysis
    if analysis_type == "Word Frequency":
        st.subheader("Word Frequency Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_n = st.slider("Number of top words to display:", 5, 50, 20)
        
        with col2:
            preprocessing_option = st.radio(
                "Text to analyze:",
                ["Preprocessed Text", "Original Text"]
            )
        
        text_to_analyze = preprocessed_text if preprocessing_option == "Preprocessed Text" else st.session_state.sample_text
        
        word_freq = get_word_frequency(text_to_analyze, top_n=top_n)
        
        if word_freq:
            st.subheader("Top Words")
            
            # Display as table
            word_freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
            st.dataframe(word_freq_df)
            
            # Display as chart
            fig = plot_word_frequency(word_freq, title=f"Top {top_n} Words Frequency")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No words found for frequency analysis.")
    
    # Sample analysis
    elif analysis_type == "Sample Analysis":
        st.subheader("Sample Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Text:")
            st.info(st.session_state.sample_text[:500] + ("..." if len(st.session_state.sample_text) > 500 else ""))
        
        with col2:
            st.write("Preprocessed Text:")
            st.info(preprocessed_text[:500] + ("..." if len(preprocessed_text) > 500 else ""))
        
        st.write("Text Statistics:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Length", len(st.session_state.sample_text))
        
        with col2:
            st.metric("Preprocessed Length", len(preprocessed_text))
        
        with col3:
            word_count = len(preprocessed_text.split())
            st.metric("Word Count", word_count)
    
    # Text statistics
    elif analysis_type == "Text Statistics":
        st.subheader("Text Statistics")
        
        # Calculate statistics
        original_text = st.session_state.sample_text
        original_words = original_text.split()
        preprocessed_words = preprocessed_text.split()
        
        original_length = len(original_text)
        original_word_count = len(original_words)
        preprocessed_word_count = len(preprocessed_words)
        avg_word_length = sum(len(word) for word in preprocessed_words) / max(1, preprocessed_word_count)
        removed_words = original_word_count - preprocessed_word_count
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Character Count", original_length)
            st.metric("Original Word Count", original_word_count)
        
        with col2:
            st.metric("Preprocessed Word Count", preprocessed_word_count)
            st.metric("Words Removed", removed_words)
        
        with col3:
            st.metric("Avg Word Length", round(avg_word_length, 2))
            st.metric("Reduction %", round(100 * (1 - preprocessed_word_count / max(1, original_word_count)), 1))
        
        # Word frequency for this view too
        st.subheader("Word Frequency")
        top_n = 15
        word_freq = get_word_frequency(preprocessed_text, top_n=top_n)
        
        if word_freq:
            fig = plot_word_frequency(word_freq, title=f"Top {top_n} Words Frequency")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
