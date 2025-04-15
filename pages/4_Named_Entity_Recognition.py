import streamlit as st
import pandas as pd
import os
import sys
import random

# Add the parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processor import extract_named_entities, load_spacy_model
from utils.visualizer import plot_entity_counts

# Page configuration
st.set_page_config(
    page_title="Named Entity Recognition - Medical Transcription NLP",
    page_icon="🏷️",
    layout="wide"
)

# Initialize SpaCy model
@st.cache_resource
def initialize_nlp():
    return load_spacy_model()

nlp = initialize_nlp()

st.title("Named Entity Recognition")
st.write("Extract and analyze named entities in medical transcriptions.")

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

# Initialize session state variables
if 'ner_sample_text' not in st.session_state:
    st.session_state.ner_sample_text = ""
if 'ner_entities' not in st.session_state:
    st.session_state.ner_entities = []

# Text selection
st.header("Text Selection")

col1, col2 = st.columns(2)

with col1:
    selected_column = st.selectbox("Select text column for entity extraction:", text_columns)

with col2:
    sample_option = st.radio(
        "Select sample option:",
        ["Random Sample", "Enter Index", "Enter Custom Text"]
    )

if sample_option == "Random Sample":
    if st.button("Get Random Sample"):
        if df.shape[0] > 0:
            sample_index = random.randint(0, df.shape[0] - 1)
            st.session_state.ner_sample_text = df.iloc[sample_index][selected_column]
            st.session_state.ner_entities = []
            st.rerun()

elif sample_option == "Enter Index":
    max_index = df.shape[0] - 1
    sample_index = st.number_input("Enter sample index:", 0, max_index, 0)
    
    if st.button("Load Sample"):
        st.session_state.ner_sample_text = df.iloc[sample_index][selected_column]
        st.session_state.ner_entities = []
        st.rerun()

elif sample_option == "Enter Custom Text":
    custom_text = st.text_area("Enter text for entity extraction:", st.session_state.ner_sample_text, height=200)
    
    if custom_text != st.session_state.ner_sample_text:
        st.session_state.ner_sample_text = custom_text
        st.session_state.ner_entities = []

# Display sample text
st.subheader("Sample Text")
if st.session_state.ner_sample_text:
    st.write(st.session_state.ner_sample_text[:1000] + ("..." if len(st.session_state.ner_sample_text) > 1000 else ""))
else:
    st.info("No sample text selected.")

# Entity extraction options
st.header("Entity Extraction Options")

entity_types = [
    "All Types",
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "NORP",
    "FAC",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "EVENT"
]

selected_entity_type = st.selectbox("Filter by entity type:", entity_types)

# Extract entities
if st.session_state.ner_sample_text and st.button("Extract Entities"):
    with st.spinner("Extracting entities..."):
        st.session_state.ner_entities = extract_named_entities(st.session_state.ner_sample_text, nlp)
    st.success(f"Extracted {len(st.session_state.ner_entities)} entities!")
    st.rerun()

# Display entity results
if st.session_state.ner_entities:
    st.header("Entity Extraction Results")
    
    # Filter entities by type if selected
    filtered_entities = st.session_state.ner_entities
    if selected_entity_type != "All Types":
        filtered_entities = [entity for entity in st.session_state.ner_entities if entity[1] == selected_entity_type]
    
    # Display entity counts
    st.subheader("Entity Counts")
    
    fig = plot_entity_counts(st.session_state.ner_entities, title="Named Entity Counts")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Display entities table
    st.subheader("Extracted Entities")
    
    if filtered_entities:
        entity_df = pd.DataFrame(filtered_entities, columns=["Entity", "Type"])
        
        # Add entity counts
        entity_counts = entity_df.groupby(["Entity", "Type"]).size().reset_index(name="Count")
        entity_counts = entity_counts.sort_values(by="Count", ascending=False)
        
        st.dataframe(entity_counts)
    else:
        st.info(f"No entities of type '{selected_entity_type}' found.")
    
    # Entity type explanation
    st.subheader("Entity Type Explanation")
    
    entity_explanations = {
        "PERSON": "People, including fictional characters",
        "ORG": "Companies, agencies, institutions, etc.",
        "GPE": "Countries, cities, states (Geo-Political Entity)",
        "LOC": "Non-GPE locations, mountain ranges, bodies of water",
        "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
        "DATE": "Absolute or relative dates or periods",
        "TIME": "Times smaller than a day",
        "MONEY": "Monetary values, including unit",
        "PERCENT": "Percentage, including %",
        "CARDINAL": "Numerals that do not fall under another type",
        "ORDINAL": "\"first\", \"second\", etc.",
        "QUANTITY": "Measurements, as of weight or distance",
        "NORP": "Nationalities or religious or political groups",
        "FAC": "Buildings, airports, highways, bridges, etc.",
        "WORK_OF_ART": "Titles of books, songs, etc.",
        "LAW": "Named documents made into laws",
        "LANGUAGE": "Any named language",
        "EVENT": "Named hurricanes, battles, wars, sports events, etc."
    }
    
    explanation_df = pd.DataFrame({
        "Entity Type": list(entity_explanations.keys()),
        "Description": list(entity_explanations.values())
    })
    
    st.dataframe(explanation_df)
    
    # Additional entity analysis
    if len(filtered_entities) > 0:
        st.subheader("Entity Context Analysis")
        
        # Get the most frequent entities
        top_entities = entity_counts.head(5)
        
        for _, row in top_entities.iterrows():
            entity = row["Entity"]
            entity_type = row["Type"]
            
            st.write(f"**{entity}** ({entity_type})")
            
            # Find original sentences containing this entity
            doc = nlp(st.session_state.ner_sample_text)
            sentences = list(doc.sents)
            
            # Find sentences containing the entity
            entity_sentences = []
            for sentence in sentences:
                if entity.lower() in sentence.text.lower():
                    entity_sentences.append(sentence.text.strip())
            
            # Display sentences (limit to 3 for readability)
            if entity_sentences:
                for i, sentence in enumerate(entity_sentences[:3]):
                    st.markdown(f"- Context {i+1}: _{sentence}_")
            else:
                st.markdown("No clear context found.")
