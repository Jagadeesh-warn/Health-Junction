import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from utils.data_loader import load_sample_data
from utils.text_processor import preprocess_text, load_spacy_model
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Set page configuration
st.set_page_config(page_title="Advanced NLP Visualizations", page_icon="📈")

st.title("Advanced NLP Visualizations")

st.write("""
This page presents advanced NLP visualizations derived from the medical transcriptions dataset,
including term frequency analysis, part-of-speech distributions, and topic modeling visualizations.
""")

# Load data
data = load_sample_data()

# Create tabs for different visualization categories
tab1, tab2, tab3 = st.tabs(["POS Tag Analysis", "Term Co-occurrence", "Topic Modeling"])

with tab1:
    st.header("Part-of-Speech Analysis")
    
    # Download required NLTK resources if needed
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        with st.spinner("Downloading required NLTK resources..."):
            nltk.download('averaged_perceptron_tagger')
    
    # Function to get POS tags from text
    @st.cache_data
    def get_pos_tags(text):
        if not isinstance(text, str):
            return []
        words = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(words)
        return tagged
    
    # Function to count POS tags
    @st.cache_data
    def count_pos_tags(tagged_text):
        pos_counts = Counter(tag for word, tag in tagged_text)
        return pos_counts
    
    # Get POS tag dictionary with descriptions
    pos_dict = {
        'CC': 'Coordinating conjunction',
        'CD': 'Cardinal number',
        'DT': 'Determiner',
        'EX': 'Existential there',
        'FW': 'Foreign word',
        'IN': 'Preposition or subordinating conjunction',
        'JJ': 'Adjective',
        'JJR': 'Adjective, comparative',
        'JJS': 'Adjective, superlative',
        'LS': 'List item marker',
        'MD': 'Modal',
        'NN': 'Noun, singular or mass',
        'NNS': 'Noun, plural',
        'NNP': 'Proper noun, singular',
        'NNPS': 'Proper noun, plural',
        'PDT': 'Predeterminer',
        'POS': 'Possessive ending',
        'PRP': 'Personal pronoun',
        'PRP$': 'Possessive pronoun',
        'RB': 'Adverb',
        'RBR': 'Adverb, comparative',
        'RBS': 'Adverb, superlative',
        'RP': 'Particle',
        'SYM': 'Symbol',
        'TO': 'to',
        'UH': 'Interjection',
        'VB': 'Verb, base form',
        'VBD': 'Verb, past tense',
        'VBG': 'Verb, gerund or present participle',
        'VBN': 'Verb, past participle',
        'VBP': 'Verb, non-3rd person singular present',
        'VBZ': 'Verb, 3rd person singular present',
        'WDT': 'Wh-determiner',
        'WP': 'Wh-pronoun',
        'WP$': 'Possessive wh-pronoun',
        'WRB': 'Wh-adverb'
    }
    
    # Select a sample for POS analysis (to prevent processing too much data)
    sample_size = min(10, len(data))
    st.info(f"Analyzing POS tags from {sample_size} sample medical transcriptions...")
    
    # Process sample data
    sample_texts = data['transcription'].head(sample_size).fillna("")
    all_pos_tags = []
    
    for text in sample_texts:
        tags = get_pos_tags(text)
        all_pos_tags.extend(tags)
    
    # Count POS tags
    pos_counts = count_pos_tags(all_pos_tags)
    
    # Create a dataframe for visualization
    pos_df = pd.DataFrame({
        'POS Tag': list(pos_counts.keys()),
        'Count': list(pos_counts.values()),
        'Description': [pos_dict.get(tag, 'Unknown') for tag in pos_counts.keys()]
    }).sort_values('Count', ascending=False)
    
    # Visualize the POS distribution
    fig = px.bar(
        pos_df.head(15),  # Show top 15 POS tags
        x='POS Tag',
        y='Count',
        color='Count',
        hover_data=['Description'],
        labels={'Count': 'Frequency', 'POS Tag': 'Part of Speech Tag'},
        title='Top 15 Parts of Speech in Medical Transcriptions',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show POS tag descriptions
    with st.expander("View POS Tag Descriptions"):
        st.dataframe(
            pos_df[['POS Tag', 'Description', 'Count']].reset_index(drop=True),
            use_container_width=True
        )
    
    # Show specific POS examples
    if all_pos_tags:
        st.subheader("Examples of Common POS Tags in Medical Text")
        
        # Get examples for top 5 POS tags
        top_tags = pos_df.head(5)['POS Tag'].tolist()
        examples = {}
        
        for tag in top_tags:
            tag_examples = [word for word, pos in all_pos_tags if pos == tag]
            unique_examples = list(set(tag_examples))[:10]  # Get up to 10 unique examples
            examples[tag] = unique_examples
        
        for tag in top_tags:
            if examples[tag]:
                st.write(f"**{tag}** ({pos_dict.get(tag, 'Unknown')}): {', '.join(examples[tag])}")

with tab2:
    st.header("Term Co-occurrence Analysis")
    
    # Function to get bigrams from text
    @st.cache_data
    def get_bigrams(text, top_n=20):
        if not isinstance(text, str):
            return []
        
        # Process text
        words = nltk.word_tokenize(text.lower())
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
        
        words = [word for word in words if word.isalpha() and word not in stop_words]
        
        # Get bigrams
        bigrams = list(nltk.bigrams(words))
        bigram_counts = Counter(bigrams)
        
        # Return top N bigrams
        return bigram_counts.most_common(top_n)
    
    # Prepare bigram data from all texts
    all_text = " ".join(data['transcription'].fillna("").tolist())
    top_bigrams = get_bigrams(all_text)
    
    # Create a dataframe for visualization
    bigram_df = pd.DataFrame(top_bigrams, columns=['Bigram', 'Count'])
    bigram_df['Bigram'] = bigram_df['Bigram'].apply(lambda x: f"{x[0]}-{x[1]}")
    
    # Visualize the top bigrams
    fig = px.bar(
        bigram_df,
        x='Count',
        y='Bigram',
        orientation='h',
        title='Top Term Co-occurrences in Medical Transcriptions',
        labels={'Count': 'Frequency', 'Bigram': 'Term Pair'},
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Create co-occurrence network visualization (simplified)
    st.subheader("Term Co-occurrence Network")
    
    # Prepare data for network visualization
    top_n_for_network = min(10, len(bigram_df))
    network_data = bigram_df.head(top_n_for_network)
    
    # Split bigrams back into source and target
    network_data[['source', 'target']] = network_data['Bigram'].str.split('-', expand=True)
    
    # Generate unique nodes
    nodes = list(set(network_data['source'].tolist() + network_data['target'].tolist()))
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Create network graph
    fig = go.Figure()
    
    # Add edges
    for _, row in network_data.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[node_indices[row['source']], node_indices[row['target']]],
                y=[0, 0],
                mode='lines',
                line=dict(width=row['Count'] / max(network_data['Count']) * 5),
                name=row['Bigram']
            )
        )
    
    # Add nodes
    for node, idx in node_indices.items():
        fig.add_trace(
            go.Scatter(
                x=[idx],
                y=[0],
                mode='markers+text',
                marker=dict(size=10),
                text=node,
                textposition="top center",
                name=node
            )
        )
    
    fig.update_layout(
        title="Term Co-occurrence Network (Simplified)",
        showlegend=False,
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("Note: This is a simplified representation of the term co-occurrence network.")

with tab3:
    st.header("Topic Modeling Visualization")
    
    # Function to display LDA topics
    def display_topics(model, feature_names, n_top_words=10):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topic_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append({
                'Topic': topic_idx + 1,
                'Terms': ", ".join(topic_terms)
            })
        return pd.DataFrame(topics)
    
    # LDA parameters selection
    st.subheader("Latent Dirichlet Allocation (LDA) Topic Model")
    
    col1, col2 = st.columns(2)
    with col1:
        n_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5, step=1)
    with col2:
        n_top_words = st.slider("Top Words per Topic", min_value=5, max_value=20, value=10, step=1)
    
    # Process data for topic modeling
    st.info("Running LDA topic modeling on the medical transcriptions...")
    
    # Prepare documents
    documents = data['transcription'].fillna("").tolist()
    
    # Create bag of words
    vectorizer = CountVectorizer(
        max_df=0.95, 
        min_df=2,
        stop_words='english'
    )
    
    # Generate bag of words
    X = vectorizer.fit_transform(documents)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Build LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
    )
    
    # Fit the model
    lda.fit(X)
    
    # Display topics
    topics_df = display_topics(lda, feature_names, n_top_words)
    
    # Show topics in a table
    st.table(topics_df)
    
    # Visualize topic-term weights
    st.subheader("Topic-Term Weights")
    
    # Get top terms for each topic
    topic_term_weights = []
    for topic_idx, topic in enumerate(lda.components_):
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            topic_term_weights.append({
                'Topic': f"Topic {topic_idx + 1}",
                'Term': feature_names[i],
                'Weight': topic[i]
            })
    
    weights_df = pd.DataFrame(topic_term_weights)
    
    # Create heatmap
    fig = px.density_heatmap(
        weights_df,
        x='Topic',
        y='Term',
        z='Weight',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Topic-Term Weight Heatmap'
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Document-Topic Distribution
    st.subheader("Document-Topic Distribution")
    
    # Transform documents to get topic distributions
    doc_topic_dist = lda.transform(X)
    
    # Get average topic distribution
    avg_topic_dist = doc_topic_dist.mean(axis=0)
    
    # Create dataframe for visualization
    topic_dist_df = pd.DataFrame({
        'Topic': [f"Topic {i+1}" for i in range(n_topics)],
        'Weight': avg_topic_dist
    })
    
    # Create topic distribution chart
    fig = px.bar(
        topic_dist_df,
        x='Topic',
        y='Weight',
        title='Average Topic Distribution Across Documents',
        color='Weight',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)