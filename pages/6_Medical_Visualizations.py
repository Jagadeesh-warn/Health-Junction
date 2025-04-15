import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from utils.data_loader import load_sample_data
from utils.text_processor import get_word_frequency, preprocess_text, extract_named_entities, load_spacy_model

st.set_page_config(page_title="Medical Visualizations", page_icon="📊")

st.title("Medical Text Visualizations")

st.write("""
This page provides various visualizations based on the medical transcriptions data, 
showing patterns and insights derived from the text analysis.
""")

# Load the sample data
data = load_sample_data()

# Create tabs for different visualization categories
tab1, tab2, tab3, tab4 = st.tabs(["Word Frequencies", "Medical Specialties", "Text Length Analysis", "Entity Visualizations"])

with tab1:
    st.header("Word Frequency Visualizations")
    
    st.subheader("Top Medical Terms")
    
    # Getting all text combined
    all_text = " ".join(data["transcription"].fillna("").tolist())
    processed_text = preprocess_text(all_text)
    
    # Extract top words
    word_freq = get_word_frequency(processed_text, top_n=30)
    
    # Create a horizontal bar chart for word frequencies
    fig = px.bar(
        x=[count for _, count in word_freq],
        y=[word for word, _ in word_freq],
        orientation='h',
        labels={'x': 'Frequency', 'y': 'Medical Term'},
        title='Most Common Medical Terms in Transcriptions',
        color=[count for _, count in word_freq],
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a word cloud if wordcloud package is available
    try:
        from wordcloud import WordCloud
        
        st.subheader("Word Cloud of Medical Terms")
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, contour_width=3, contour_color='steelblue')
        
        # Generate word cloud
        word_dict = dict(word_freq)
        wordcloud.generate_from_frequencies(word_dict)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
        
    except ImportError:
        st.info("WordCloud package is not installed. Install it to view the word cloud visualization.")

with tab2:
    st.header("Medical Specialty Analysis")
    
    # Count the medical specialties
    specialty_counts = data["medical_specialty"].value_counts().reset_index()
    specialty_counts.columns = ["Specialty", "Count"]
    
    # Display specialty distribution
    fig = px.pie(
        specialty_counts, 
        values='Count', 
        names='Specialty',
        title='Distribution of Medical Specialties',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart of specialties
    fig2 = px.bar(
        specialty_counts.sort_values('Count', ascending=False), 
        x='Specialty', 
        y='Count',
        title='Count of Records by Medical Specialty',
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig2.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Text Length Analysis")
    
    # Calculate text lengths
    data['text_length'] = data['transcription'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Histogram of text lengths
    fig = px.histogram(
        data, 
        x='text_length',
        nbins=20,
        title='Distribution of Transcription Lengths',
        labels={'text_length': 'Text Length (characters)'},
        color_discrete_sequence=['#0068C9']
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Text length by specialty
    specialty_length = data.groupby('medical_specialty')['text_length'].mean().sort_values(ascending=False).reset_index()
    specialty_length.columns = ['Specialty', 'Average Length']
    
    fig2 = px.bar(
        specialty_length,
        x='Specialty',
        y='Average Length',
        title='Average Transcription Length by Medical Specialty',
        color='Average Length',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig2.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.header("Named Entity Visualizations")
    
    # Load spacy model
    with st.spinner("Loading NLP model..."):
        nlp = load_spacy_model()
    
    # Sample limit to prevent overloading
    sample_limit = min(20, len(data))
    
    # Extract entities from a subset of transcriptions
    st.info(f"Analyzing entities from {sample_limit} sample transcriptions...")
    
    all_entities = []
    for text in data['transcription'].head(sample_limit):
        if isinstance(text, str):
            entities = extract_named_entities(text, nlp)
            all_entities.extend(entities)
    
    # Count entity types
    entity_types = [entity_type for _, entity_type in all_entities]
    entity_counts = Counter(entity_types)
    
    # Create a bar chart for entity counts
    entity_df = pd.DataFrame({
        'Entity Type': list(entity_counts.keys()),
        'Count': list(entity_counts.values())
    }).sort_values('Count', ascending=False)
    
    fig = px.bar(
        entity_df,
        x='Entity Type',
        y='Count',
        title='Named Entity Types in Medical Transcriptions',
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a table with sample entities
    st.subheader("Sample Entities Extracted")
    
    # Get sample entities for each type
    entity_samples = {}
    for entity, entity_type in all_entities:
        if entity_type not in entity_samples:
            entity_samples[entity_type] = []
        if entity not in entity_samples[entity_type] and len(entity_samples[entity_type]) < 5:
            entity_samples[entity_type].append(entity)
    
    # Convert to dataframe for display
    samples_list = []
    for entity_type, entities in entity_samples.items():
        samples_list.append({
            'Entity Type': entity_type,
            'Examples': ', '.join(entities)
        })
    
    samples_df = pd.DataFrame(samples_list)
    st.dataframe(samples_df, use_container_width=True)