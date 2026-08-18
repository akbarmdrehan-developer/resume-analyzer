import re
import spacy
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cache the model load so the app runs smoothly
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_resume_info(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email = re.findall(email_pattern, text)
    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phone = re.findall(phone_pattern, text)
    
    skill_bank = ['python', 'java', 'c++', 'javascript', 'sql', 'machine learning', 
                  'deep learning', 'react', 'html', 'css', 'git', 'aws', 'data analysis']
    extracted_skills = [skill.title() for skill in skill_bank if skill in text.lower()]
            
    return {"Email": email if email else "Not Found", 
            "Phone": phone if phone else "Not Found", 
            "Skills": extracted_skills}

def calculate_match_score(resume_text, job_desc_text):
    documents = [resume_text, job_desc_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity) * 100, 2)

# --- Streamlit UI Design ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📊", layout="centered")
st.title("📊 AI Resume Analyzer & Parser")
st.markdown("### *7th Semester Engineering Project*")
st.write("---")

uploaded_file = st.file_uploader("Upload Resume (PDF format only)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here", height=150, placeholder="Looking for a Python developer skilled in SQL...")

if st.button("🚀 Analyze and Match Resume"):
    if uploaded_file and job_description:
        with st.spinner("Analyzing text patterns..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            info = extract_resume_info(resume_text)
            match_score = calculate_match_score(resume_text, job_description)
            
            # Match Score Card
            st.metric(label="🎯 Job Match Score", value=f"{match_score}%")
            
            # Columns layout
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📧 Contact Details")
                st.markdown(f"**Email:** {info['Email']}")
                st.markdown(f"**Phone:** {info['Phone']}")
            with col2:
                st.subheader("🛠️ Extracted Skills")
                if info['Skills']:
                    st.success(", ".join(info['Skills']))
                else:
                    st.warning("No skills matched from the basic tech bank.")
    else:
        st.error("⚠️ Please provide both the resume PDF and the job description text.")
