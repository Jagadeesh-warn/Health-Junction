
Health Junction: NLP-Based Medical Question Answering System
============================================================

<img width="600" alt="image" src="https://github.com/user-attachments/assets/dfbb6252-6d68-4d14-96a5-991af1bfe612" />


A natural language processing (NLP) powered medical chatbot that helps general users understand symptoms, diseases, and possible conditions. The system extracts insights from real clinical transcriptions and medical literature to generate medically relevant responses.

Features
--------
- Multisource Knowledge Retrieval: Extracts insights from clinical notes and OCR-processed medical books.
- User-friendly Chat Interface: Built using Streamlit for interactive symptom-based query handling.
- Clinical Language Processing: Uses ClinicalBERT and SciSpaCy for accurate medical term identification and semantic similarity.
- Knowledge Graph (Experimental): Conceptual mapping of entities to specialties for interpretability.

Project Architecture
--------------------

User Query
   │
   ▼
Text Preprocessing (nltk, regex, lemmatization)
   │
   ├──> Named Entity Recognition (SciSpaCy / spaCy)
   ├──> Embeddings (TF-IDF / ClinicalBERT)
   │
   ▼
Similarity Search
   ├──> Clinical Notes (MTSamples)
   └──> Medical Book (OCR content)
   │
   ▼
Response Generation (Insights + Summary + Disclaimer)
   │
   ▼
Streamlit Chat Interface

Dataset Sources
---------------
- Clinical Notes: MTSamples Dataset on Kaggle
- Medical Book: Extracted from “Professional Guide to Diseases” using pytesseract

Tech Stack
----------
Component                | Tool/Library
-------------------------|------------------------------
UI                      | Streamlit
NLP Pipeline            | NLTK, SciSpaCy, spaCy
Embedding Models        | TF-IDF, ClinicalBERT
OCR                     | Pytesseract
Visualization (Optional)| Yellowbrick, Matplotlib
Data                    | CSV (notes) + OCR (book)

Sample Input/Output
-------------------
User: I am suffering from fever and cough

Response:
1. Clinical Note Match: “68-year-old man with 3 days of greenish sputum”
2. Book Reference: “Page 267: Influenza diagnosis and treatment”
3. Summary: Possible conditions - flu, pneumonia, infection

Installation & Setup
--------------------
1. Clone the Repository:
   git clone https://github.com/Jagadeesh-warn/health-junction.git
   cd health-junction

2. Create Virtual Environment:
   python3 -m venv env
   source env/bin/activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Download NLTK Data:
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')

5. Run the App:
   streamlit run app.py

Project Structure
-----------------
health-junction/
├── app.py
├── pages/
├── utils/
│   ├── data_loader.py
│   ├── text_processor.py
│   └── visualizer.py
├── data/
│   ├── mtsamples.csv
│   └── book_extracted.csv
├── README.txt
└── requirements.txt


License
-------
MIT License
