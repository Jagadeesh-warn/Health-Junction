import pandas as pd
import streamlit as st
import os
import tempfile
import kagglehub
import time
import random

@st.cache_data
def load_sample_data():
    """
    Load a small sample of the medical transcriptions dataset for demonstration purposes.
    Uses cached data to avoid reloading on every rerun.
    
    Returns:
        pandas.DataFrame: Sample of medical transcriptions
    """
    try:
        # Create a small sample dataframe for initial viewing
        sample_data = {
            'description': [
                "Patient presents with chest pain and shortness of breath.",
                "Follow-up visit for diabetes management.",
                "MRI results show mild degenerative changes.",
                "Patient reports headaches and dizziness.",
                "Annual physical examination with no significant findings."
            ],
            'medical_specialty': [
                "Cardiovascular / Pulmonary",
                "Endocrinology",
                "Orthopedic",
                "Neurology",
                "General Medicine"
            ],
            'sample_name': [
                "Sample A",
                "Sample B",
                "Sample C",
                "Sample D",
                "Sample E"
            ],
            'transcription': [
                "SUBJECTIVE: Patient is a 45-year-old male presenting with sudden onset of chest pain and shortness of breath while exercising. Pain is described as pressure-like, radiating to the left arm. Patient reports associated diaphoresis and nausea. No prior history of cardiac issues. Family history significant for early MI in father. OBJECTIVE: Vital signs: BP 145/90, HR 92, RR 20, O2 sat 95% on room air. Physical exam reveals mild distress. Heart with regular rate and rhythm, no murmurs. Lungs clear to auscultation bilaterally. ECG shows ST elevation in leads II, III, aVF. ASSESSMENT: Acute inferior wall myocardial infarction. PLAN: Immediate cardiology consult. Aspirin 325mg given. Initiating protocol for cardiac catheterization.",
                "ASSESSMENT: Type 2 diabetes mellitus with suboptimal control. Patient is a 62-year-old female with 10-year history of T2DM, currently on metformin 1000mg BID and glipizide 10mg daily. Recent HbA1c 8.2%, up from 7.5% three months ago. Patient reports increased thirst and urination over past month. Denies vision changes or neuropathic symptoms. OBJECTIVE: Weight 182 lbs (up 3 lbs from last visit). BP 138/82, HR 74. Foot exam with normal sensation, no ulcerations. PLAN: Increase metformin to 1000mg TID. Start Jardiance 10mg daily. Diabetes education referral. Lab work including comprehensive metabolic panel, lipid panel, HbA1c in 3 months.",
                "MRI FINDINGS: Lumbar spine MRI demonstrates mild degenerative disc disease at L4-L5 and L5-S1 with small posterior disc bulges, not impinging on the thecal sac or nerve roots. No spinal stenosis identified. Mild facet arthropathy at L4-L5. No fractures or lesions. Alignment is preserved. Paraspinal soft tissues unremarkable. IMPRESSION: 1. Mild degenerative changes of the lower lumbar spine without significant neural compression. 2. Mild facet arthropathy at L4-L5. RECOMMENDATION: Conservative management with physical therapy and anti-inflammatory medications. Follow-up as needed if symptoms worsen.",
                "CHIEF COMPLAINT: Headaches and dizziness for 2 weeks. HISTORY: Patient is a 36-year-old female with new onset headaches described as throbbing, primarily frontal, associated with photophobia and occasional nausea. Patient also reports intermittent dizziness, particularly with position changes. No head trauma. No fever. PHYSICAL EXAMINATION: Vitals stable. HEENT: Normal fundoscopic exam. No papilledema. Neurological exam nonfocal. DIAGNOSTIC STUDIES: CT head without contrast: No acute intracranial abnormalities. CBC, CMP within normal limits. ASSESSMENT: 1. Migraine headaches with vestibular symptoms 2. Rule out vestibular neuritis PLAN: 1. Sumatriptan 50mg PRN for acute headaches 2. Vestibular testing 3. Follow-up in 2 weeks",
                "ANNUAL PHYSICAL EXAMINATION: 52-year-old female, generally healthy, presents for routine checkup. No significant complaints. Maintains active lifestyle with regular exercise. Non-smoker, occasional alcohol use. Up-to-date on screenings. PHYSICAL EXAM: Vitals: BP 120/76, HR 68, RR 16, Temp 98.2F, BMI 23.4. General: Well-appearing, in no distress. HEENT: Normal. Cardiopulmonary: Regular rate and rhythm, no murmurs. Clear lung fields. Abdomen: Soft, non-tender. Extremities: No edema. Skin: No concerning lesions. LABS: CBC, CMP, lipid panel within normal limits. ASSESSMENT: Healthy adult. Appropriate age-related screenings current. PLAN: Continue current health maintenance. Flu vaccine administered. Return in one year for routine follow-up."
            ],
            'keywords': [
                "chest pain, shortness of breath, MI, myocardial infarction, ECG, ST elevation",
                "diabetes mellitus, metformin, HbA1c, glycemic control, Jardiance",
                "MRI, lumbar spine, disc disease, degenerative changes, facet arthropathy",
                "headache, dizziness, migraine, vestibular, neurological, sumatriptan",
                "annual exam, physical, routine, screening, preventive care"
            ]
        }
        
        return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error creating sample data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_kaggle_dataset():
    """
    Load the medical transcriptions dataset from Kaggle.
    Uses cached data to avoid reloading on every rerun.
    
    Returns:
        str: Path to the dataset files
    """
    try:
        with st.spinner("Downloading dataset from Kaggle..."):
            path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
        return path
    except Exception as e:
        st.error(f"Error downloading dataset from Kaggle: {e}")
        return None

def load_dataset_from_path(path):
    """
    Load the dataset from the specified path.
    
    Args:
        path (str): Path to the dataset files
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        if path and os.path.exists(path):
            # Get all CSV files in the path
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            if not csv_files:
                st.warning(f"No CSV files found in {path}")
                return None
            
            # Load the first CSV file
            file_path = os.path.join(path, csv_files[0])
            df = pd.read_csv(file_path)
            return df
        else:
            st.warning("Invalid path provided.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def load_dataset_from_upload(uploaded_file):
    """
    Load dataset from an uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    try:
        if uploaded_file is not None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load the dataset from the temporary file
            df = pd.read_csv(tmp_path)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
            
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None

def get_data_info(df):
    """
    Get information about the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset
        
    Returns:
        dict: Dataset information
    """
    if df is None or df.empty:
        return None
    
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample": df.head(5)
    }
    
    return info
