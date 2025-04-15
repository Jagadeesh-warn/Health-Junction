import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np

def plot_word_frequency(word_freq, title="Top Words Frequency"):
    """
    Plot word frequency using Plotly.
    
    Args:
        word_freq (list): List of (word, frequency) tuples
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if not word_freq:
        return None
    
    words, frequencies = zip(*word_freq)
    
    fig = px.bar(
        x=words,
        y=frequencies,
        labels={'x': 'Words', 'y': 'Frequency'},
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Words",
        yaxis_title="Frequency",
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def plot_pos_distribution(pos_tags, title="POS Tags Distribution"):
    """
    Plot part-of-speech tags distribution using Plotly.
    
    Args:
        pos_tags (dict): Dictionary of POS tags and their counts
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if not pos_tags:
        return None
    
    pos_df = pd.DataFrame.from_dict(pos_tags, orient='index', columns=['count'])
    pos_df = pos_df.reset_index().rename(columns={'index': 'pos'})
    pos_df = pos_df.sort_values(by='count', ascending=False)
    
    fig = px.pie(
        pos_df,
        values='count',
        names='pos',
        title=title
    )
    
    return fig

def plot_entity_counts(entities, title="Named Entity Counts"):
    """
    Plot named entity counts using Plotly.
    
    Args:
        entities (list): List of (entity, label) tuples
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if not entities:
        return None
    
    entity_labels = [label for _, label in entities]
    label_counts = Counter(entity_labels)
    
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    fig = px.bar(
        x=labels,
        y=counts,
        labels={'x': 'Entity Type', 'y': 'Count'},
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Entity Type",
        yaxis_title="Count",
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def plot_specialty_distribution(df, specialty_column="medical_specialty", title="Medical Specialty Distribution"):
    """
    Plot medical specialty distribution using Plotly.
    
    Args:
        df (pandas.DataFrame): Dataset containing medical specialty information
        specialty_column (str): Name of the column containing specialty information
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or specialty_column not in df.columns:
        return None
    
    specialty_counts = df[specialty_column].value_counts().reset_index()
    specialty_counts.columns = ['Specialty', 'Count']
    
    # Limit to top 15 specialties for readability
    if len(specialty_counts) > 15:
        top_specialties = specialty_counts.head(14)
        other_count = specialty_counts.iloc[14:]['Count'].sum()
        other_row = pd.DataFrame({'Specialty': ['Other'], 'Count': [other_count]})
        specialty_counts = pd.concat([top_specialties, other_row], ignore_index=True)
    
    fig = px.pie(
        specialty_counts,
        values='Count',
        names='Specialty',
        title=title
    )
    
    return fig

def plot_text_length_distribution(df, text_column="transcription", title="Text Length Distribution"):
    """
    Plot text length distribution using Plotly.
    
    Args:
        df (pandas.DataFrame): Dataset containing text data
        text_column (str): Name of the column containing text data
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if df is None or df.empty or text_column not in df.columns:
        return None
    
    # Calculate text lengths
    text_lengths = df[text_column].str.len()
    
    fig = px.histogram(
        text_lengths,
        nbins=50,
        labels={'value': 'Text Length (characters)', 'count': 'Frequency'},
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Text Length (characters)",
        yaxis_title="Frequency"
    )
    
    return fig

def plot_topic_distribution(topic_distribution, title="Topic Distribution"):
    """
    Plot topic distribution using Plotly.
    
    Args:
        topic_distribution (list): List of (topic_id, weight) tuples
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if not topic_distribution:
        return None
    
    topic_ids, weights = zip(*topic_distribution)
    topic_names = [f"Topic {tid}" for tid in topic_ids]
    
    fig = px.pie(
        names=topic_names,
        values=weights,
        title=title
    )
    
    return fig
