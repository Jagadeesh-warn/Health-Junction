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

# Check if required columns exist
required_columns = ['transcription', 'medical_specialty']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
    st.info("Please make sure your dataset contains the necessary columns for visualization.")
    st.stop()

# Create tabs for different visualization categories
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Word Frequencies", 
    "Medical Specialties", 
    "Text Length Analysis", 
    "Entity Visualizations",
    "Knowledge Graph",
    "POS Analysis"
])

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
    
with tab5:
    st.header("Medical Knowledge Graph Visualization")
    
    try:
        import networkx as nx
        
        st.subheader("Medical Specialty - Term Knowledge Graph")
        
        # Sample limit to prevent overloading
        sample_limit = min(50, len(data))
        
        with st.spinner("Building knowledge graph..."):
            # Create knowledge graph using NetworkX
            knowledge_graph = nx.Graph()
            
            # Process sample of data
            sample_data = data.head(sample_limit)
            
            # Add nodes for medical specialties
            specialties = sample_data['medical_specialty'].unique()
            for specialty in specialties:
                knowledge_graph.add_node(specialty, node_type='specialty')
            
            # Add nodes for terms
            # First get word frequencies from the sample
            all_sample_text = " ".join(sample_data["transcription"].fillna("").tolist())
            processed_text = preprocess_text(all_sample_text)
            word_freq = dict(get_word_frequency(processed_text, top_n=50))
            
            # Add word nodes
            for word, count in word_freq.items():
                if count >= 5:  # Adjust threshold as needed
                    knowledge_graph.add_node(word, node_type='term', count=count)
            
            # Add edges between specialties and terms
            for idx, row in sample_data.iterrows():
                specialty = row['medical_specialty']
                text = row['transcription']
                if isinstance(text, str):
                    # Process text to get words
                    processed = preprocess_text(text)
                    words = processed.split()
                    
                    # Add edges for words that meet threshold
                    for word in set(words):
                        if word in word_freq and word_freq[word] >= 5:
                            if knowledge_graph.has_edge(specialty, word):
                                knowledge_graph[specialty][word]['weight'] += 1
                            else:
                                knowledge_graph.add_edge(specialty, word, weight=1)
            
            # Create the visualization
            if len(knowledge_graph.nodes()) > 0:
                # Get node types for coloring
                node_types = nx.get_node_attributes(knowledge_graph, 'node_type')
                
                # Create lists for visualization
                color_map = []
                for node in knowledge_graph.nodes():
                    if node_types.get(node) == 'specialty':
                        color_map.append('lightblue')
                    else:
                        color_map.append('lightgreen')
                
                # Adjust figure parameters for Streamlit
                plt.figure(figsize=(10, 8))  # Smaller figure for Streamlit
                
                # Create layout
                pos = nx.spring_layout(knowledge_graph, k=0.3, seed=42)
                
                # Draw the graph
                nx.draw(
                    knowledge_graph, 
                    pos, 
                    with_labels=True, 
                    node_size=500,  
                    node_color=color_map,
                    font_size=8, 
                    font_color="black", 
                    edge_color="gray", 
                    width=0.5,
                    alpha=0.8
                )
                
                plt.title("Medical Specialty - Term Knowledge Graph")
                st.pyplot(plt)
                
                # Graph statistics
                st.subheader("Knowledge Graph Statistics")
                st.write(f"Number of nodes: {len(knowledge_graph.nodes())}")
                st.write(f"Number of edges: {len(knowledge_graph.edges())}")
                st.write(f"Number of specialties: {len([n for n, d in knowledge_graph.nodes(data=True) if d.get('node_type') == 'specialty'])}")
                st.write(f"Number of medical terms: {len([n for n, d in knowledge_graph.nodes(data=True) if d.get('node_type') == 'term'])}")
                
            else:
                st.warning("Not enough data to create a knowledge graph. Try increasing the sample size.")
    
    except ImportError:
        st.error("NetworkX is required for knowledge graph visualization. Please install it using 'pip install networkx'.")
    except Exception as e:
        st.error(f"Error building knowledge graph: {e}")

with tab6:
    st.header("Part-of-Speech Analysis")
    
    # Load spacy model
    with st.spinner("Loading NLP model..."):
        nlp = load_spacy_model()
    
    st.subheader("POS Distribution in Medical Texts")
    
    # Sample limit to prevent overloading
    sample_limit = min(20, len(data))
    
    # Analyze POS in a subset of transcriptions
    st.info(f"Analyzing part-of-speech distribution from {sample_limit} sample transcriptions...")
    
    # Function to get POS distribution
    def get_pos_distribution(texts, nlp):
        pos_counts = Counter()
        for text in texts:
            if isinstance(text, str):
                doc = nlp(text[:5000])  # Limit length to prevent memory issues
                for token in doc:
                    pos_counts[token.pos_] += 1
        return pos_counts
    
    # Get POS distribution
    sample_texts = data['transcription'].head(sample_limit).tolist()
    pos_counts = get_pos_distribution(sample_texts, nlp)
    
    # Convert to DataFrame for visualization
    pos_df = pd.DataFrame({
        'POS Tag': list(pos_counts.keys()),
        'Count': list(pos_counts.values())
    }).sort_values('Count', ascending=False)
    
    # Create a bar chart for POS distribution
    fig = px.bar(
        pos_df,
        x='POS Tag',
        y='Count',
        title='Part-of-Speech Distribution in Medical Transcriptions',
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # POS Tag Explanation
    st.subheader("POS Tag Explanations")
    pos_explanations = {
        'NOUN': 'Nouns (names of people, places, things)',
        'VERB': 'Verbs (actions, states)',
        'ADJ': 'Adjectives (descriptive words)',
        'ADV': 'Adverbs (modify verbs, adjectives)',
        'ADP': 'Adpositions (prepositions, postpositions)',
        'DET': 'Determiners (articles, demonstratives)',
        'PRON': 'Pronouns (I, you, he, she, etc.)',
        'CONJ': 'Conjunctions (and, or, but)',
        'NUM': 'Numerals (numbers)',
        'PART': 'Particles',
        'INTJ': 'Interjections (oh, wow)',
        'PROPN': 'Proper nouns (names of specific entities)',
        'PUNCT': 'Punctuation',
        'SYM': 'Symbols',
        'X': 'Other'
    }
    
    explanations_df = pd.DataFrame({
        'POS Tag': list(pos_explanations.keys()),
        'Explanation': list(pos_explanations.values())
    })
    
    st.dataframe(explanations_df, use_container_width=True)
    
    # Get most common words by POS type
    st.subheader("Common Words by POS Category")
    
    selected_pos = st.selectbox("Select POS category to view common words:", 
                              ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'])
    
    # Function to extract words with specific POS tag
    def get_common_words_by_pos(texts, nlp, pos_tag, limit=15):
        word_counts = Counter()
        for text in texts:
            if isinstance(text, str):
                doc = nlp(text[:5000])  # Limit length to prevent memory issues
                for token in doc:
                    if token.pos_ == pos_tag and not token.is_stop and len(token.text) > 2:
                        word_counts[token.text.lower()] += 1
        return word_counts.most_common(limit)
    
    # Get common words for selected POS
    common_words = get_common_words_by_pos(sample_texts, nlp, selected_pos)
    
    # Create DataFrame for visualization
    if common_words:
        words_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
        
        # Create bar chart
        fig = px.bar(
            words_df,
            x='Word',
            y='Count',
            title=f'Most Common {selected_pos} Words in Medical Transcriptions',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No common {selected_pos} words found in the sample.")