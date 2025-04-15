import pandas as pd
import streamlit as st
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string

# Ensure NLTK resources are downloaded
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")

# Load SpaCy model
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    """
    Load the SpaCy language model.
    
    Args:
        model_name (str): Name of the SpaCy model to load
        
    Returns:
        spacy.Language: Loaded SpaCy model
    """
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        # If model is not installed, try downloading it
        st.info(f"Downloading SpaCy model {model_name}...")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
        return nlp
    except Exception as e:
        st.error(f"Error loading SpaCy model: {e}")
        return None

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Preprocess text for NLP analysis.
    
    Args:
        text (str): Text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to lemmatize words
        
    Returns:
        str: Preprocessed text
    """
    download_nltk_resources()
    
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def get_word_frequency(text, top_n=20):
    """
    Get word frequency from text.
    
    Args:
        text (str): Text to analyze
        top_n (int): Number of top words to return
        
    Returns:
        list: List of (word, frequency) tuples
    """
    if not text or not isinstance(text, str):
        return []
    
    download_nltk_resources()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    
    # Count word frequency
    word_freq = Counter(filtered_tokens)
    
    # Get top N words
    top_words = word_freq.most_common(top_n)
    
    return top_words

def analyze_sentence_structure(text, nlp=None):
    """
    Analyze sentence structure using SpaCy.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.Language): SpaCy model
        
    Returns:
        dict: Sentence structure analysis
    """
    if not text or not isinstance(text, str):
        return {}
    
    if nlp is None:
        nlp = load_spacy_model()
    
    if nlp is None:
        return {}
    
    doc = nlp(text)
    
    analysis = {
        "sentences": len(list(doc.sents)),
        "pos_tags": Counter([token.pos_ for token in doc]),
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "noun_chunks": [chunk.text for chunk in doc.noun_chunks]
    }
    
    return analysis

def extract_named_entities(text, nlp=None):
    """
    Extract named entities from text using SpaCy.
    
    Args:
        text (str): Text to analyze
        nlp (spacy.Language): SpaCy model
        
    Returns:
        list: List of (entity, label) tuples
    """
    if not text or not isinstance(text, str):
        return []
    
    if nlp is None:
        nlp = load_spacy_model()
    
    if nlp is None:
        return []
    
    doc = nlp(text)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return entities
