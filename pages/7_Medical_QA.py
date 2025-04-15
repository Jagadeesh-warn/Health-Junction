import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import load_sample_data
from utils.text_processor import load_spacy_model

st.set_page_config(page_title="Medical Question Answering", page_icon="❓")

st.title("Medical Question Answering System")

st.write("""
This page provides a question answering system for medical queries. You can ask medical questions,
and the system will search through the medical transcriptions to find the most relevant information.
""")

# Download NLTK resources if needed
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Initialize resources
download_nltk_resources()

# Load the sample data
data = load_sample_data()

# Define the MedicalChatbot class
class MedicalChatbot:
    def __init__(self, clinical_notes_df):
        """
        Initialize the medical chatbot
        
        Parameters:
        - clinical_notes_df: DataFrame of clinical notes
        """
        # Store input DataFrame
        self.clinical_notes = clinical_notes_df
        
        # TF-IDF vectorizer for searching
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Build TF-IDF index
        self.build_knowledge_index()
        
        # Define medical entity keywords
        self._initialize_keywords()
    
    def _initialize_keywords(self):
        """Initialize medical keyword dictionaries for entity extraction"""
        # Basic custom keywords
        self.disease_keywords = ['disease', 'cancer', 'flu', 'infection', 'hypertension',
                                'diabetes', 'asthma', 'pneumonia', 'bronchitis', 'arthritis',
                                'disorder', 'syndrome', 'illness', 'condition']
        
        self.symptom_keywords = ['fever', 'cough', 'headache', 'pain', 'nausea',
                                'shortness of breath', 'dizziness', 'fatigue', 'weakness',
                                'swelling', 'vomiting', 'ache', 'rash', 'discomfort']
        
        self.treatment_keywords = ['treatment', 'medication', 'drug', 'therapy', 'surgery',
                                  'antibiotics', 'prescription', 'dose', 'vaccine', 'injection',
                                  'remedy', 'cure', 'regimen', 'administered']
        
        self.recovery_keywords = ['days', 'weeks', 'months', 'recovery', 'healing',
                                 'recuperation', 'convalescence', 'rehabilitation',
                                 'prognosis', 'outlook', 'follow-up']
        
        # Keywords for advice and warning signs
        self.advice_keywords = ['recommend', 'advised', 'should', 'consider', 'suggested',
                               'avoid', 'maintain', 'increase', 'decrease', 'monitor',
                               'consult', 'follow-up', 'important to', 'best to',
                               'ideal', 'beneficial']
        
        self.warning_keywords = ['warning', 'danger', 'concerning', 'emergency', 'urgent',
                                'critical', 'immediate attention', 'severe', 'worsening',
                                'deterioration', 'fatal', 'life-threatening', 'call 911',
                                'go to hospital', 'alert', 'red flag', 'caution', 'watch for',
                                'concerning sign', 'alarming', 'contact doctor if']
        
        # Modal verbs for advice detection
        self.modal_verbs = ['should', 'must', 'can', 'could', 'may', 'might', 'need']
        
        # Action verbs for advice detection
        self.action_verbs = ['take', 'avoid', 'increase', 'decrease', 'monitor', 'schedule']
        
        # Patient-directed phrases
        self.patient_phrases = ['patient should', 'patients should', 'recommend', 'advised to', 'home care']
    
    def build_knowledge_index(self):
        """Build searchable TF-IDF index for clinical notes."""
        # Check if transcription column exists
        if 'transcription' not in self.clinical_notes.columns:
            st.error("The dataset does not contain a 'transcription' column.")
            return
            
        # Clinical notes index
        self.vectors = self.vectorizer.fit_transform(
            self.clinical_notes['transcription'].fillna('')
        )
    
    def extract_medical_entities_from_text(self, text):
        """
        Extract diseases, symptoms, treatments, recovery info, advice and warning signs
        from any text using keyword matching and pattern recognition.
        """
        # Initialize entity buckets
        entities = {
            'diseases': set(),
            'symptoms': set(),
            'treatments': set(),
            'recovery': set(),
            'advice': set(),
            'warning_signs': set()
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Keyword-based checks
        self._extract_entities_by_keywords(text_lower, entities)
        
        # Extract advice and warning signs using more sophisticated methods
        sentences = sent_tokenize(text)
        self._extract_advice_from_sentences(sentences, entities)
        self._extract_warning_signs_from_sentences(sentences, entities)
        
        return entities
    
    def _extract_entities_by_keywords(self, text_lower, entities):
        """Extract basic medical entities using keyword matching"""
        # Check for disease keywords
        for kw in self.disease_keywords:
            if kw in text_lower:
                entities['diseases'].add(kw)
                
        # Check for symptom keywords
        for kw in self.symptom_keywords:
            if kw in text_lower:
                entities['symptoms'].add(kw)
                
        # Check for treatment keywords
        for kw in self.treatment_keywords:
            if kw in text_lower:
                entities['treatments'].add(kw)
                
        # Check for recovery keywords
        for kw in self.recovery_keywords:
            if kw in text_lower:
                entities['recovery'].add(kw)
    
    def _extract_advice_from_sentences(self, sentences, entities):
        """
        Extract medical advice from sentences using keyword matching
        and sentence pattern recognition.
        """
        for sent in sentences:
            sent_lower = sent.lower()
            
            # Check for advice keywords
            has_advice_keyword = any(kw in sent_lower for kw in self.advice_keywords)
            
            # Check for modal verbs
            has_modal_verb = any(f" {verb} " in f" {sent_lower} " for verb in self.modal_verbs)
            
            if has_advice_keyword or has_modal_verb:
                # Look for patient-directed advice
                if any(phrase in sent_lower for phrase in self.patient_phrases):
                    entities['advice'].add(sent.strip())
                # Look for general medical recommendations
                elif any(f" {verb} " in f" {sent_lower} " for verb in self.action_verbs):
                    entities['advice'].add(sent.strip())
    
    def _extract_warning_signs_from_sentences(self, sentences, entities):
        """
        Extract medical warning signs from sentences using keyword matching
        and sentence pattern recognition.
        """
        for sent in sentences:
            sent_lower = sent.lower()
            
            # Check for warning patterns
            if any(kw in sent_lower for kw in self.warning_keywords):
                entities['warning_signs'].add(sent.strip())
                
            # Look for conditional warnings ("If X, then seek medical attention")
            if ('if' in sent_lower and any(phrase in sent_lower for phrase in
                                         ['seek', 'call', 'consult', 'emergency', 'doctor',
                                          'physician', 'hospital', 'immediate', 'urgent'])):
                entities['warning_signs'].add(sent.strip())
                
            # Look for mentions of severe symptoms
            if any(phrase in sent_lower for phrase in
                  ['severe', 'worsening', 'persistent', 'does not improve']):
                if any(symptom in sent_lower for symptom in
                      ['pain', 'fever', 'vomiting', 'bleeding', 'breath', 'consciousness']):
                    entities['warning_signs'].add(sent.strip())
    
    def search_clinical_notes(self, query):
        """
        Retrieve top 3 matching clinical notes based on TF-IDF similarity.
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        top_indices = similarities.argsort()[-3:][::-1]
        return self.clinical_notes.iloc[top_indices]
    
    def generate_response(self, query):
        """
        Generate a structured response that includes:
          1) Insights from Clinical Notes
          2) Diagnosed Condition, Symptoms, Treatment, Estimated Recovery
          3) Personal Medical Advice
          4) Warning Signs to Watch For
        """
        # 1) Extract entities from the user query
        user_entities = self.extract_medical_entities_from_text(query)
        
        # 2) Get top clinical notes
        clinical_matches = self.search_clinical_notes(query)
        
        # 3) Aggregate entities from matched clinical notes
        for _, note in clinical_matches.iterrows():
            note_entities = self.extract_medical_entities_from_text(note['transcription'])
            for entity_type in user_entities:
                user_entities[entity_type].update(note_entities[entity_type])
        
        # 4) Format the final multi-part summary
        response_text = self.format_response(query, user_entities, clinical_matches)
        return response_text
    
    def format_response(self, query, aggregated_entities, clinical_matches):
        """
        Build a four-part output:
          1) Insights from Clinical Notes
          2) Overall Summary (Diagnosed Condition, Symptoms, Treatment, Recovery)
          3) Personal Medical Advice
          4) Warning Signs to Watch For
        """
        # Convert sets to comma-separated strings
        disease_str = ", ".join(sorted(aggregated_entities['diseases'])) or "Not Identified"
        symptoms_str = ", ".join(sorted(aggregated_entities['symptoms'])) or "Not Mentioned"
        treatment_str = ", ".join(sorted(aggregated_entities['treatments'])) or "Consult a Doctor"
        recovery_str = ", ".join(sorted(aggregated_entities['recovery'])) or "Varies"
        
        # 1) Insights from Clinical Notes
        notes_section = ["### 1. Insights from Clinical Notes"]
        if clinical_matches.empty:
            notes_section.append("No relevant clinical notes found.")
        else:
            for i, (_, note) in enumerate(clinical_matches.iterrows(), 1):
                # For each note, extract a snippet (first 150 chars)
                text = note['transcription']
                snippet = text[:150] + "..." if len(text) > 150 else text
                notes_section.append(f"**Match {i}:** {snippet}")
        
        # 2) Overall Medical Summary
        summary_section = ["### 2. Medical Summary"]
        summary_section.append(f"**Possible Condition(s):** {disease_str}")
        summary_section.append(f"**Common Symptoms:** {symptoms_str}")
        summary_section.append(f"**Treatment Options:** {treatment_str}")
        summary_section.append(f"**Recovery Information:** {recovery_str}")
        
        # 3) Personal Medical Advice
        advice_section = ["### 3. Medical Advice"]
        advice = list(aggregated_entities['advice'])
        if advice:
            for adv in advice[:5]:  # Limit to top 5 advice items
                advice_section.append(f"- {adv}")
        else:
            advice_section.append("No specific medical advice found. Please consult a healthcare professional.")
        
        # 4) Warning Signs
        warning_section = ["### 4. Warning Signs to Watch For"]
        warnings = list(aggregated_entities['warning_signs'])
        if warnings:
            for warn in warnings[:5]:  # Limit to top 5 warnings
                warning_section.append(f"- {warn}")
        else:
            warning_section.append("No specific warning signs mentioned. If symptoms worsen, seek medical attention.")
        
        # Build the complete response
        all_sections = [
            "\n".join(notes_section),
            "\n".join(summary_section),
            "\n".join(advice_section),
            "\n".join(warning_section),
            "\n\n**Disclaimer:** This information is for educational purposes only and should not " +
            "replace professional medical advice. Always consult a healthcare professional for " +
            "medical diagnoses and treatment."
        ]
        
        return "\n\n".join(all_sections)

# Create an instance of the MedicalChatbot
@st.cache_resource
def initialize_chatbot():
    return MedicalChatbot(data)

chatbot = initialize_chatbot()

# User input
st.subheader("Ask a Medical Question")
st.write("Enter your medical question below. The system will search for relevant information in the medical transcriptions.")

query = st.text_area("Your medical question:", height=100, 
                    placeholder="Example: What are the symptoms of diabetes and how is it treated?")

if st.button("Get Answer"):
    if query:
        with st.spinner("Processing your question..."):
            response = chatbot.generate_response(query)
            st.markdown(response)
    else:
        st.warning("Please enter a question.")

# Show some example questions
with st.expander("Example Questions"):
    st.markdown("""
    Here are some example questions you can try:
    
    * What are the symptoms of pneumonia?
    * How is hypertension treated?
    * What is the recovery period for surgery?
    * What are the warning signs of a heart attack?
    * How is diabetes diagnosed and managed?
    """)

# Disclaimer
st.markdown("""
---
**Important Disclaimer:** This is a demonstration system using a limited dataset.
The information provided should not be used for medical diagnosis or treatment.
Always consult a qualified healthcare professional for medical advice.
""")