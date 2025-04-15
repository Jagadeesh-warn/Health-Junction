import streamlit as st
import pandas as pd
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Add the parent directory to path to allow importing from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processor import preprocess_text
from utils.visualizer import plot_topic_distribution

# Page configuration
st.set_page_config(
    page_title="Topic Modeling - Medical Transcription NLP",
    page_icon="📊",
    layout="wide"
)

st.title("Topic Modeling")
st.write("Discover topics in medical transcription corpus using unsupervised learning.")

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

# Initialize session state variables
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None

# Topic modeling options
st.header("Topic Modeling Options")

col1, col2 = st.columns(2)

with col1:
    selected_column = st.selectbox("Select text column for topic modeling:", text_columns)
    
    modeling_algorithm = st.selectbox(
        "Select topic modeling algorithm:",
        ["Latent Dirichlet Allocation (LDA)", "Non-Negative Matrix Factorization (NMF)"]
    )
    
    vectorization_method = st.selectbox(
        "Select vectorization method:",
        ["CountVectorizer", "TF-IDF Vectorizer"]
    )

with col2:
    num_topics = st.slider("Number of topics:", 2, 20, 5)
    
    max_features = st.slider("Maximum number of features:", 100, 10000, 1000)
    
    # Display info about the chosen algorithm
    if modeling_algorithm == "Latent Dirichlet Allocation (LDA)":
        st.info("""
        **LDA** works well for discovering abstract topics in documents.
        It assumes each document is a mixture of topics and each topic is a mixture of words.
        """)
    else:
        st.info("""
        **NMF** often produces more coherent topics than LDA.
        It works well with TF-IDF vectorization for extracting meaningful topics.
        """)

# Sample size selection
sample_size = st.slider(
    "Select sample size for topic modeling (larger samples take longer to process):",
    min_value=50,
    max_value=min(1000, df.shape[0]),
    value=min(200, df.shape[0]),
    step=50
)

# Preprocess and train topic model
if st.button("Run Topic Modeling"):
    with st.spinner("Preprocessing text and training topic model..."):
        # Sample data
        if df.shape[0] > sample_size:
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
        
        # Preprocess text
        texts = sample_df[selected_column].fillna("").astype(str).tolist()
        preprocessed_texts = [preprocess_text(text) for text in texts]
        
        # Vectorize text
        if vectorization_method == "CountVectorizer":
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2
            )
        else:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2
            )
        
        # Transform text to document-term matrix
        dtm = vectorizer.fit_transform(preprocessed_texts)
        
        # Train topic model
        if modeling_algorithm == "Latent Dirichlet Allocation (LDA)":
            model = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
        else:
            model = NMF(
                n_components=num_topics,
                random_state=42,
                max_iter=200
            )
        
        transformed_data = model.fit_transform(dtm)
        
        # Store results in session state
        st.session_state.topic_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.feature_names = vectorizer.get_feature_names_out()
        st.session_state.transformed_data = transformed_data
        
    st.success("Topic modeling completed successfully!")
    st.rerun()

# Display topic modeling results
if st.session_state.topic_model is not None:
    st.header("Topic Modeling Results")
    
    # Display top words for each topic
    st.subheader("Topics and Top Words")
    
    model = st.session_state.topic_model
    feature_names = st.session_state.feature_names
    
    # Function to display top words for each topic
    def display_topics(model, feature_names, n_top_words=10):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                "Topic": topic_idx,
                "Top Words": ", ".join(top_words)
            })
        return pd.DataFrame(topics)
    
    topics_df = display_topics(model, feature_names)
    st.dataframe(topics_df)
    
    # Document-topic distribution visualization
    st.subheader("Document-Topic Distribution")
    
    # Select a random document to visualize
    doc_idx = st.slider("Select document index:", 0, len(st.session_state.transformed_data)-1, 0)
    
    # Get topic distribution for selected document
    topic_distribution = [(i, dist) for i, dist in enumerate(st.session_state.transformed_data[doc_idx])]
    topic_distribution.sort(key=lambda x: x[1], reverse=True)
    
    # Plot topic distribution
    fig = plot_topic_distribution(topic_distribution, title="Topic Distribution for Selected Document")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Display original document
    if df.shape[0] > doc_idx:
        st.subheader("Original Document")
        st.write(df.iloc[doc_idx][selected_column])
    
    # Topic interpretation
    st.subheader("Topic Interpretation")
    
    # Get the dominant topic for each document
    dominant_topics = st.session_state.transformed_data.argmax(axis=1)
    
    # Count documents per topic
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    
    # Create a bar chart of document counts per topic
    topic_counts_df = pd.DataFrame({
        'Topic': topic_counts.index,
        'Document Count': topic_counts.values
    })
    
    st.bar_chart(topic_counts_df.set_index('Topic'))
    
    # Topic coherence explanation
    st.info("""
    **Topic Coherence** measures how semantically coherent the discovered topics are.
    
    For well-formed topics:
    - Words within a topic should be semantically related
    - Documents should tend to focus on a small number of topics
    - Topics should be distinct from each other
    
    Adjust the number of topics and features to find the most coherent topics.
    """)
    
    # Allow downloading topic model results
    if st.button("Prepare Topic Results CSV"):
        # Create a DataFrame with document-topic distributions
        result_df = pd.DataFrame(st.session_state.transformed_data)
        result_df.columns = [f"Topic_{i}" for i in range(num_topics)]
        
        # Add dominant topic column
        result_df["Dominant_Topic"] = dominant_topics
        
        # Convert to CSV
        csv = result_df.to_csv(index=False)
        
        st.download_button(
            label="Download Topic Results CSV",
            data=csv,
            file_name="topic_modeling_results.csv",
            mime="text/csv"
        )
else:
    st.info("Run topic modeling to see results.")
