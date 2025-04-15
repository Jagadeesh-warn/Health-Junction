import streamlit as st
import pandas as pd
import os
import sys

# Add the parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processor import analyze_sentence_structure, load_spacy_model
from utils.visualizer import plot_pos_distribution

# Page configuration
st.set_page_config(
    page_title="NLP Pipeline - Medical Transcription NLP",
    page_icon="🔬",
    layout="wide"
)

# Initialize SpaCy model
@st.cache_resource
def initialize_nlp():
    return load_spacy_model()

nlp = initialize_nlp()

st.title("NLP Pipeline Analysis")
st.write("Process text through comprehensive NLP pipelines for detailed analysis.")

# Check if dataset is loaded
if 'dataset' not in st.session_state or st.session_state.dataset is None:
    st.warning("No dataset loaded. Please go to the Data Exploration page to load your data.")
    st.stop()

if nlp is None:
    st.error("Failed to load SpaCy model. Please check your installation.")
    st.stop()

df = st.session_state.dataset

# Determine appropriate text columns
text_columns = [col for col in df.columns if df[col].dtype == 'object']

if not text_columns:
    st.error("No text columns found in the dataset.")
    st.stop()

# Initialize session state variables for text selection
if 'nlp_sample_option' not in st.session_state:
    st.session_state.nlp_sample_option = "random"
if 'nlp_sample_index' not in st.session_state:
    st.session_state.nlp_sample_index = 0
if 'nlp_sample_text' not in st.session_state:
    st.session_state.nlp_sample_text = ""
if 'nlp_analysis_result' not in st.session_state:
    st.session_state.nlp_analysis_result = None

# Text selection
st.header("Text Selection")

col1, col2 = st.columns(2)

with col1:
    selected_column = st.selectbox("Select text column for NLP analysis:", text_columns)

with col2:
    sample_option = st.radio(
        "Select sample option:",
        ["Random Sample", "Enter Index", "Enter Custom Text"]
    )

if sample_option == "Random Sample":
    st.session_state.nlp_sample_option = "random"
    if st.button("Get Random Sample"):
        if df.shape[0] > 0:
            st.session_state.nlp_sample_index = df.sample(1).index[0]
            st.session_state.nlp_sample_text = df.iloc[st.session_state.nlp_sample_index][selected_column]
            st.session_state.nlp_analysis_result = None
            st.rerun()

elif sample_option == "Enter Index":
    st.session_state.nlp_sample_option = "index"
    max_index = df.shape[0] - 1
    sample_index = st.number_input("Enter sample index:", 0, max_index, st.session_state.nlp_sample_index)
    
    if sample_index != st.session_state.nlp_sample_index:
        st.session_state.nlp_sample_index = sample_index
        st.session_state.nlp_sample_text = df.iloc[sample_index][selected_column]
        st.session_state.nlp_analysis_result = None
        st.rerun()

elif sample_option == "Enter Custom Text":
    st.session_state.nlp_sample_option = "custom"
    custom_text = st.text_area("Enter text for analysis:", st.session_state.nlp_sample_text, height=200)
    
    if custom_text != st.session_state.nlp_sample_text:
        st.session_state.nlp_sample_text = custom_text
        st.session_state.nlp_analysis_result = None
        st.rerun()

# Display sample text
st.subheader("Sample Text")
if st.session_state.nlp_sample_text:
    st.write(st.session_state.nlp_sample_text[:1000] + ("..." if len(st.session_state.nlp_sample_text) > 1000 else ""))
else:
    st.info("No sample text selected.")

# NLP Pipeline options
st.header("NLP Pipeline Options")

pipeline_option = st.selectbox(
    "Select NLP analysis type:",
    ["Sentence Structure Analysis", "Part-of-Speech Tagging", "Dependency Parsing"]
)

# Process text through NLP pipeline
if st.session_state.nlp_sample_text and st.button("Run NLP Analysis"):
    with st.spinner("Processing text through NLP pipeline..."):
        st.session_state.nlp_analysis_result = analyze_sentence_structure(st.session_state.nlp_sample_text, nlp)
    st.success("Analysis complete!")
    st.rerun()

# Display analysis results
if st.session_state.nlp_analysis_result:
    st.header("Analysis Results")
    
    result = st.session_state.nlp_analysis_result
    
    if pipeline_option == "Sentence Structure Analysis":
        st.subheader("Sentence Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Sentences", result["sentences"])
            st.metric("Number of Noun Chunks", len(result["noun_chunks"]))
        
        with col2:
            st.metric("Number of Entities", len(result["entities"]))
            pos_count = sum(result["pos_tags"].values())
            st.metric("Number of Tokens", pos_count)
        
        st.subheader("Noun Chunks")
        if result["noun_chunks"]:
            chunks_df = pd.DataFrame(result["noun_chunks"], columns=["Noun Chunk"])
            st.dataframe(chunks_df)
        else:
            st.info("No noun chunks found.")
    
    elif pipeline_option == "Part-of-Speech Tagging":
        st.subheader("Part-of-Speech Tags")
        
        pos_tags = result["pos_tags"]
        
        if pos_tags:
            # Create a mapping of POS tag codes to descriptions
            pos_descriptions = {
                "ADJ": "Adjective",
                "ADP": "Adposition",
                "ADV": "Adverb",
                "AUX": "Auxiliary",
                "CONJ": "Conjunction",
                "CCONJ": "Coordinating conjunction",
                "DET": "Determiner",
                "INTJ": "Interjection",
                "NOUN": "Noun",
                "NUM": "Numeral",
                "PART": "Particle",
                "PRON": "Pronoun",
                "PROPN": "Proper noun",
                "PUNCT": "Punctuation",
                "SCONJ": "Subordinating conjunction",
                "SYM": "Symbol",
                "VERB": "Verb",
                "X": "Other"
            }
            
            # Create DataFrame with descriptions
            pos_df = pd.DataFrame({
                'POS Tag': list(pos_tags.keys()),
                'Description': [pos_descriptions.get(tag, tag) for tag in pos_tags.keys()],
                'Count': list(pos_tags.values())
            }).sort_values(by='Count', ascending=False)
            
            st.dataframe(pos_df)
            
            # Plot POS distribution
            fig = plot_pos_distribution(pos_tags, title="Part-of-Speech Distribution")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No POS tags found.")
    
    elif pipeline_option == "Dependency Parsing":
        st.subheader("Dependency Parsing")
        
        # For dependency parsing, we need to run the SpaCy model directly
        doc = nlp(st.session_state.nlp_sample_text[:1000])  # Limit to 1000 chars for visualization
        
        # Prepare data for visualization
        tokens = [token.text for token in doc]
        deps = [token.dep_ for token in doc]
        heads = [token.head.text for token in doc]
        
        # Create DataFrame
        deps_df = pd.DataFrame({
            'Token': tokens,
            'Dependency': deps,
            'Head': heads
        })
        
        st.dataframe(deps_df)
        
        # Display visual explanation
        st.subheader("Dependency Parsing Explanation")
        st.markdown("""
        Dependency parsing analyzes the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads.
        
        Common dependency types:
        - **nsubj**: Nominal subject
        - **dobj**: Direct object
        - **det**: Determiner
        - **prep**: Preposition
        - **pobj**: Object of preposition
        - **amod**: Adjectival modifier
        - **advmod**: Adverbial modifier
        - **aux**: Auxiliary verb
        - **conj**: Conjunction
        - **cc**: Coordinating conjunction
        """)
        
        # Suggest using an external visualization tool
        st.info("For a visual representation of dependency parsing, consider using tools like displaCy in a full SpaCy implementation.")
