import streamlit as st
import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import load_sample_data
from utils.text_processor import load_spacy_model, preprocess_text

st.set_page_config(page_title="Medical Question Answering", page_icon="❓")

st.title("Medical Question Answering System")

st.write("""
This advanced medical question answering system searches through both clinical notes 
and medical book content to provide comprehensive responses to your medical queries.
""")

# Download NLTK resources if needed
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Initialize resources
download_nltk_resources()

# Load the sample data
data = load_sample_data()

# Define the Medical Book content - simulating a medical reference database
# This would typically be loaded from a file, but for demonstration purposes, we'll create some sample content
@st.cache_data
def load_medical_book_content():
    medical_book = [
        {
            "page": 267,
            "content": """Type Signs and symptoms Diagnosis Treatment Influenza Cough (initially Chest X-ray: (prognosis poor nonproductive; later, diffuse bilateraleven with purulent sputum), bronchopneumoniatreatment; 30 marked cyanosis, radiating from hilusmortality) dyspnea, high fever, white blood cells count:chills, substemal pain normal to slightlyand discomfort, moist elevatedcrackles, frontal Sputum smears:headache, and no specificmyalgia organisms Death results fromcardiopulmonarycollapse"""
        },
        {
            "page": 77,
            "content": """monitoring for signs of cardiac compression or cardiac tamponade,possible complications of pericardial effusion (Signs include decreasedblood pressure, increased CVP, and pulsus paradoxus. Because cardiac tamponade is life-threatening, it is crucial to recognize the warningsigns and report them immediately."""
        },
        {
            "page": 580,
            "content": """Pediatric respiratory infections: Bronchiolitis, pneumonia, and croup aresome of the most common respiratory issues affecting children. Typically,symptoms include coughing, fever, and varying degrees of respiratorydistress. Treatment depends on the specific diagnosis but may includehydration, oxygen therapy, and sometimes antibiotics (for bacterial cases)."""
        },
        {
            "page": 960,
            "content": """SYMPTOMS OF HIV INFECTIONMemory loss, disorientation,inability to think clearly Persistent headaches High fever White patches on tongue Swollen lymphnodes in neck,armpits, and groin Heavy night sweats Loss ofappetite Severeweight loss Cryptospo-ridiosis:severediarrhea,weight loss Chronicdiarrhea AIDS-RELATED ILLNESSES ANDOPPORTUNISTIC INFECTIONSCryptococcal meningitis: inflammationin and around brain and centralnervous system (CNS)Toxoplasmosis encephalitis:most common opportunistic infection of CNS,causes brain lesions"""
        },
        {
            "page": 421,
            "content": """Diabetes management requires careful monitoring of blood glucose levels.Treatment typically involves a combination of lifestyle modifications(diet and exercise), oral medications, and/or insulin therapy. Patientswith Type 1 diabetes require insulin, while Type 2 may be managed withoral medications initially. Warning signs of poor glycemic control includeextreme thirst, frequent urination, blurred vision, and in severe cases,diabetic ketoacidosis which can be life-threatening."""
        },
        {
            "page": 156,
            "content": """Hypertension is often called the 'silent killer' because it typically has nosymptoms until significant damage has occurred. Treatment includes lifestylechanges (reduced sodium intake, regular exercise, stress management) andmedications such as diuretics, ACE inhibitors, ARBs, calcium channelblockers, and beta-blockers. Regular monitoring is essential, and patientsmust understand the importance of medication adherence even when feelingwell."""
        }
    ]
    return pd.DataFrame(medical_book)

medical_book_df = load_medical_book_content()

# Define the Enhanced MedicalChatbot class
class EnhancedMedicalChatbot:
    def __init__(self, clinical_notes_df, medical_book_df):
        """
        Initialize the enhanced medical chatbot
        
        Parameters:
        - clinical_notes_df: DataFrame of clinical notes
        - medical_book_df: DataFrame of medical book content
        """
        # Store input DataFrames
        self.clinical_notes = clinical_notes_df
        self.medical_book = medical_book_df
        
        # TF-IDF vectorizers for searching different content sources
        self.clinical_vectorizer = TfidfVectorizer(stop_words='english')
        self.book_vectorizer = TfidfVectorizer(stop_words='english')
        
        # Build TF-IDF indices
        self.build_knowledge_indices()
        
        # Define medical entity keywords
        self._initialize_keywords()
        
        # Load SpaCy model for entity recognition
        self.nlp = load_spacy_model()
    
    def _initialize_keywords(self):
        """Initialize comprehensive medical keyword dictionaries for entity extraction"""
        # Diseases and conditions
        self.disease_keywords = [
            'disease', 'cancer', 'flu', 'infection', 'hypertension', 'diabetes', 
            'asthma', 'pneumonia', 'bronchitis', 'arthritis', 'disorder', 'syndrome', 
            'illness', 'condition', 'virus', 'bacterial', 'inflammation', 'tumor',
            'fracture', 'cardiovascular', 'pulmonary', 'neurological', 'hiv', 'aids',
            'covid', 'copd', 'emphysema', 'appendicitis', 'meningitis', 'hepatitis'
        ]
        
        # Symptoms
        self.symptom_keywords = [
            'fever', 'cough', 'headache', 'pain', 'nausea', 'shortness of breath', 
            'dizziness', 'fatigue', 'weakness', 'swelling', 'vomiting', 'ache', 
            'rash', 'discomfort', 'chills', 'sweats', 'dyspnea', 'chest pain', 
            'sore throat', 'runny nose', 'congestion', 'wheezing', 'confusion', 
            'diarrhea', 'constipation', 'bleeding', 'bruising', 'palpitations',
            'tachycardia', 'bradycardia', 'hypertension', 'hypotension'
        ]
        
        # Treatments
        self.treatment_keywords = [
            'treatment', 'medication', 'drug', 'therapy', 'surgery', 'antibiotics', 
            'prescription', 'dose', 'vaccine', 'injection', 'remedy', 'cure', 
            'regimen', 'administered', 'protocol', 'procedure', 'management', 
            'intervention', 'rehabilitation', 'physical therapy', 'chemotherapy', 
            'radiation', 'dialysis', 'transplant', 'oxygen', 'ventilator', 
            'intubation', 'sutures', 'cast', 'prednisone', 'steroid'
        ]
        
        # Recovery
        self.recovery_keywords = [
            'days', 'weeks', 'months', 'recovery', 'healing', 'recuperation', 
            'convalescence', 'rehabilitation', 'prognosis', 'outlook', 'follow-up',
            'outcome', 'discharge', 'post-operative', 'restoration', 'improvement',
            'recovery period', 'expected', 'typical timeline', 'monitoring'
        ]
        
        # Advice keywords
        self.advice_keywords = [
            'recommend', 'advised', 'should', 'consider', 'suggested', 'avoid', 
            'maintain', 'increase', 'decrease', 'monitor', 'consult', 'follow-up', 
            'important to', 'best to', 'ideal', 'beneficial', 'encourage', 
            'discourage', 'essential', 'critical', 'needed', 'necessary', 
            'required', 'instruction', 'direction', 'guidance', 'indication'
        ]
        
        # Warning signs
        self.warning_keywords = [
            'warning', 'danger', 'concerning', 'emergency', 'urgent', 'critical', 
            'immediate attention', 'severe', 'worsening', 'deterioration', 'fatal', 
            'life-threatening', 'call 911', 'go to hospital', 'alert', 'red flag', 
            'caution', 'watch for', 'concerning sign', 'alarming', 'contact doctor if',
            'seek help', 'immediate care', 'without delay', 'complication', 'crisis',
            'acute', 'ambulance', 'ER', 'emergency room', 'lethal', 'death'
        ]
        
        # Modal verbs for advice detection
        self.modal_verbs = ['should', 'must', 'can', 'could', 'may', 'might', 'need']
        
        # Action verbs for advice detection
        self.action_verbs = [
            'take', 'avoid', 'increase', 'decrease', 'monitor', 'schedule', 'apply',
            'administer', 'consume', 'stop', 'start', 'continue', 'follow', 'check',
            'measure', 'track', 'report', 'call', 'visit', 'consult', 'seek'
        ]
        
        # Patient-directed phrases
        self.patient_phrases = [
            'patient should', 'patients should', 'recommend', 'advised to', 'home care',
            'self-care', 'self-management', 'at-home', 'manage symptoms', 'lifestyle',
            'diet', 'exercise', 'rest', 'fluid intake', 'medication regimen'
        ]
    
    def build_knowledge_indices(self):
        """Build searchable TF-IDF indices for clinical notes and medical book content."""
        # Verify clinical notes data
        if 'transcription' not in self.clinical_notes.columns:
            st.error("The clinical notes dataset does not contain a 'transcription' column.")
            return
            
        # Verify medical book data
        if 'content' not in self.medical_book.columns:
            st.error("The medical book dataset does not contain a 'content' column.")
            return
        
        # Build clinical notes index
        self.clinical_vectors = self.clinical_vectorizer.fit_transform(
            self.clinical_notes['transcription'].fillna('')
        )
        
        # Build medical book index
        self.book_vectors = self.book_vectorizer.fit_transform(
            self.medical_book['content'].fillna('')
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
        
        # Also use SpaCy for entity recognition
        if len(text) > 0:
            self._extract_entities_with_spacy(text, entities)
        
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
    
    def _extract_entities_with_spacy(self, text, entities):
        """Use SpaCy for more advanced entity recognition"""
        try:
            # Process text with SpaCy (limit length for efficiency)
            doc = self.nlp(text[:10000])
            
            # Extract medical entities based on entity types
            for ent in doc.ents:
                # Map SpaCy's entity types to our categories
                if ent.label_ == "DISEASE" or ent.label_ == "CONDITION":
                    entities['diseases'].add(ent.text.lower())
                elif ent.label_ == "SYMPTOM":
                    entities['symptoms'].add(ent.text.lower())
                elif ent.label_ == "TREATMENT" or ent.label_ == "PROCEDURE":
                    entities['treatments'].add(ent.text.lower())
        except Exception as e:
            # Fail silently, just use keyword matching
            pass
    
    def _extract_advice_from_sentences(self, sentences, entities):
        """Extract medical advice from sentences using keyword matching and pattern recognition."""
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
        """Extract medical warning signs from sentences using keyword matching and pattern recognition."""
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
    
    def search_clinical_notes(self, query, top_n=3):
        """Retrieve top matching clinical notes based on TF-IDF similarity."""
        query_vector = self.clinical_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.clinical_vectors)[0]
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.05]
    
    def search_medical_book(self, query, top_n=3):
        """Retrieve top matching content from medical book based on TF-IDF similarity."""
        query_vector = self.book_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.book_vectors)[0]
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.05]
    
    def format_clinical_note(self, idx, score):
        """Format a clinical note for display."""
        note = self.clinical_notes.iloc[idx]
        text = note['transcription']
        specialty = note.get('medical_specialty', 'Unknown Specialty')
        
        # Create a brief description from the beginning of the note
        first_line = next((line for line in text.split('\n') if line.strip()), "No description available")
        description = first_line[:150] + "..." if len(first_line) > 150 else first_line
        
        # Get a relevant excerpt (first 200 chars)
        excerpt = text[:200] + "..." if len(text) > 200 else text
        
        return f"- **Description**: {description} \n  Specialty: {specialty}\n  Excerpt: {excerpt}"
    
    def format_book_content(self, idx, score):
        """Format medical book content for display."""
        book_entry = self.medical_book.iloc[idx]
        page = book_entry.get('page', 'Unknown')
        content = book_entry['content']
        
        # Format the content for display (first 200 chars)
        excerpt = content[:200] + "..." if len(content) > 200 else content
        
        return f"- **Page {page}**: {excerpt}"
    
    def generate_response(self, query):
        """
        Generate a comprehensive structured response that includes:
          1) Insights from Clinical Notes
          2) Insights from Medical Book
          3) Consolidated Medical Summary
          4) Personal Medical Advice
          5) Warning Signs to Watch For
        """
        # Extract entities from the user query
        query_entities = self.extract_medical_entities_from_text(query)
        
        # Search both data sources
        clinical_matches = self.search_clinical_notes(query, top_n=3)
        book_matches = self.search_medical_book(query, top_n=3)
        
        # Aggregate all relevant content for entity extraction
        all_content = query + "\n"
        
        # Add clinical notes content
        clinical_formatted = []
        for idx, score in clinical_matches:
            note = self.clinical_notes.iloc[idx]
            all_content += note['transcription'] + "\n"
            clinical_formatted.append(self.format_clinical_note(idx, score))
        
        # Add medical book content
        book_formatted = []
        for idx, score in book_matches:
            book_entry = self.medical_book.iloc[idx]
            all_content += book_entry['content'] + "\n"
            book_formatted.append(self.format_book_content(idx, score))
        
        # Extract all medical entities from aggregated content
        aggregated_entities = self.extract_medical_entities_from_text(all_content)
        
        # Format the comprehensive response
        response_text = self.format_comprehensive_response(
            query, 
            aggregated_entities, 
            clinical_formatted, 
            book_formatted
        )
        
        return response_text
    
    def format_comprehensive_response(self, query, entities, clinical_matches, book_matches):
        """
        Build a comprehensive five-part output:
          1) Insights from Clinical Notes
          2) Insights from Medical Book
          3) Consolidated Medical Summary
          4) Personal Medical Advice
          5) Warning Signs to Watch For
        """
        # Convert sets to comma-separated strings
        disease_str = ", ".join(sorted(entities['diseases'])) or "Not Identified"
        symptoms_str = ", ".join(sorted(entities['symptoms'])) or "Not Mentioned"
        treatment_str = ", ".join(sorted(entities['treatments'])) or "Consult a Doctor"
        recovery_str = ", ".join(sorted(entities['recovery'])) or "Varies"
        
        # 1) Insights from Clinical Notes
        notes_section = ["**1. Insights from Clinical Notes**"]
        if not clinical_matches:
            notes_section.append("No relevant clinical notes found.")
        else:
            notes_section.extend(clinical_matches)
        
        # 2) Insights from Medical Book
        book_section = ["**2. Insights from Book**"]
        if not book_matches:
            book_section.append("No relevant medical book content found.")
        else:
            book_section.extend(book_matches)
        
        # 3) Consolidated Medical Summary
        summary_section = ["**3. Consolidated Medical Summary**"]
        summary_section.append(f"**Diagnosed Condition(s)**: {disease_str}")
        summary_section.append(f"**Symptoms Identified**: {symptoms_str}")
        summary_section.append(f"**Possible Treatments**: {treatment_str}")
        summary_section.append(f"**Estimated Recovery**: {recovery_str}")
        
        # 4) Personal Medical Advice
        advice_section = ["**4. Personal Medical Advice**"]
        advice = list(entities['advice'])
        if advice:
            for adv in advice[:5]:  # Limit to top 5 advice items
                advice_section.append(f"- {adv}")
        else:
            advice_section.append("No specific medical advice found. Please consult a healthcare professional.")
        
        # 5) Warning Signs
        warning_section = ["**5. Warning Signs to Watch For**"]
        warnings = list(entities['warning_signs'])
        if warnings:
            for warn in warnings[:5]:  # Limit to top 5 warnings
                warning_section.append(f"- {warn}")
        else:
            warning_section.append("No specific warning signs mentioned. If symptoms worsen, seek immediate medical attention.")
        
        # Build the complete response
        all_sections = [
            "\n".join(notes_section),
            "\n".join(book_section),
            "\n".join(summary_section),
            "\n".join(advice_section),
            "\n".join(warning_section),
            "\n\n***Disclaimer: This information is for reference only and based on content from our database. " +
            "Always consult a healthcare professional for accurate diagnosis, treatment, and personalized advice. " +
            "If you experience any warning signs, seek immediate medical attention.***"
        ]
        
        return "\n\n".join(all_sections)

# Create an instance of the Enhanced MedicalChatbot
@st.cache_resource
def initialize_enhanced_chatbot():
    return EnhancedMedicalChatbot(data, medical_book_df)

chatbot = initialize_enhanced_chatbot()

# User input section with improved UI
st.subheader("Ask a Medical Question")
st.write("""
Enter your medical question below to search across both clinical notes and medical reference materials.
The system will provide a comprehensive answer with insights from multiple sources.
""")

query = st.text_area("Your medical question:", height=100, 
                    placeholder="Example: What are the symptoms of pneumonia and how is it treated?")

col1, col2 = st.columns([1, 3])
with col1:
    submit_button = st.button("Get Answer", type="primary")

if submit_button:
    if query:
        with st.spinner("Analyzing your question and searching medical resources..."):
            response = chatbot.generate_response(query)
            st.markdown(response)
    else:
        st.warning("Please enter a question.")

# Show some example questions with better formatting
with st.expander("Example Questions"):
    st.markdown("""
    Try asking specific medical questions such as:
    
    * What are the symptoms and treatment for pneumonia?
    * What are the warning signs of a heart attack?
    * How is diabetes diagnosed and managed?
    * What are the treatment options for hypertension?
    * What are the symptoms of COVID-19 infection?
    * How long is the recovery period after surgery?
    * What should I know about antibiotics for a respiratory infection?
    """)

# Clear and professional disclaimer
st.markdown("""
---
**Important Medical Disclaimer:** This system is for educational and informational purposes only. 
It analyzes a limited dataset of medical information and is not a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
with any questions regarding a medical condition.
""")