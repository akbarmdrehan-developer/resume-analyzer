import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_name(text):
    # Splits document text into clean lines to capture the candidate name line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Not Found"
    
    # Common headers to filter out from top-level metadata lines
    corrupt_words = {'resume', 'cv', 'curriculum', 'vitae', 'page', 'summary', 'profile'}
    
    for line in lines[:4]:
        # Filter line arrays that contain digits, emails, or generic keywords
        if (len(line) < 30 and 
            not any(char.isdigit() for char in line) and 
            '@' not in line and 
            line.lower() not in corrupt_words):
            return re.sub(r'\s+', ' ', line)
            
    return "Not Found"

def extract_resume_info(text):
    # Extract Email using Regex
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_list = re.findall(email_pattern, text)
    
    # Extract Phone Numbers
    phone_pattern = r'\b(?:\+?\d{1,3}[-. \s]?)?\(?\d{3}\)?[-. \s]?\d{3}[-. \s]?\d{4}\b'
    phone_list = re.findall(phone_pattern, text)
    
    # Advanced Software Developer Skill Matching Dictionary
    skill_bank = [
        'java', 'python', 'c++', 'javascript', 'sql', 'mysql', 'mongodb',
        'data structures', 'algorithms', 'object-oriented programming', 'oop',
        'html', 'css', 'restful apis', 'rest api', 'git', 'github',
        'problem-solving', 'analytical thinking', 'communication', 
        'teamwork', 'collaboration', 'adaptability', 'willingness to learn', 'time management'
    ]
    
    lowered_text = text.lower()
    extracted_skills = []
    
    for skill in skill_bank:
        if skill in ['c++', 'problem-solving']:
            pattern = re.escape(skill)
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            
        if re.search(pattern, lowered_text):
            # Clean presentation name formatting transformations
            if skill in ['java', 'python', 'javascript', 'mysql', 'mongodb', 'github']:
                display_name = skill.title()
            elif skill in ['sql', 'html', 'css', 'git', 'oop']:
                display_name = skill.upper()
            elif skill == 'c++':
                display_name = 'C++'
            elif skill in ['restful apis', 'rest api']:
                display_name = 'RESTful APIs'
            elif skill == 'object-oriented programming':
                display_name = 'Object-Oriented Programming (OOP)'
            else:
                display_name = skill.title()
                
            extracted_skills.append(display_name)
            
    return {
        "Name": extract_name(text),
        "Email": email_list[0].strip() if email_list else "Not Found", 
        "Phone": phone_list[0].strip() if phone_list else "Not Found", 
        "Skills": list(set(extracted_skills))
    }

def recommend_courses(resume_skills, job_desc_text):
    # Complete course mapping catalog for Software Developer options
    course_bank = {
        'Java': ['Java Programming and Software Engineering Fundamentals (Coursera)', 'Java Masterclass (Udemy)'],
        'Python': ['Python for Everybody Specialization (Coursera)', 'Complete Python Bootcamp (Udemy)'],
        'C++': ['Coding in C++ (edX)', 'Beginning C++ Programming (Udemy)'],
        'Javascript': ['JavaScript: The Advanced Concepts (Udemy)', 'Modern JavaScript Track (Educative)'],
        'SQL': ['SQL for Data Science (Coursera)', 'The Complete SQL Bootcamp (Udemy)'],
        'Mysql': ['The Ultimate MySQL Bootcamp (Udemy)', 'MySQL Database Track'],
        'Mongodb': ['MongoDB - The Complete Developer\'s Guide (Udemy)', 'MongoDB University Tracks'],
        'Data Structures': ['Data Structures and Algorithms Specialization (Coursera)', 'Mastering DSA (Udemy)'],
        'Algorithms': ['Algorithms Specialization by Stanford (Coursera)', 'Intro to Algorithms (MIT Track)'],
        'Object-Oriented Programming (OOP)': ['Object Oriented Programming in Java/C++ (Udemy)'],
        'HTML': ['Introduction to HTML5 (Coursera)', 'Web Design for Beginners (Udemy)'],
        'CSS': ['Advanced CSS and Sass (Udemy)', 'CSS - The Complete Guide (Udemy)'],
        'RESTful APIs': ['API Design and Fundamentals (Google/Coursera)', 'REST API Design Track'],
        'Git': ['Version Control with Git (Coursera)', 'Git & GitHub Masterclass (Udemy)'],
        'Github': ['GitHub Ultimate: Master Git and GitHub (Udemy)'],
        'Problem-Solving': ['Creative Problem Solving & Decision Making (Coursera)'],
        'Analytical Thinking': ['Critical Thinking and Problem Solving (edX)'],
        'Communication': ['Improving Communication Skills (Coursera)'],
        'Teamwork': ['Teamwork Skills & Collaborating Effectively (Coursera)'],
        'Collaboration': ['High-Performance Collaboration (Coursera)'],
        'Adaptability': ['Developing Adaptability & Resilience (LinkedIn Learning)'],
        'Willingness To Learn': ['Learning How to Learn by Barbara Oakley (Coursera)'],
        'Time Management': ['Work Smarter, Not Harder: Time Management (Coursera)']
    }
    
    lowered_jd = " " + " ".join(job_desc_text.lower().split()) + " "
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    # Synonym backup routing tags
    for s in resume_skills:
        if "object-oriented" in s.lower() or "oop" in s.lower():
            resume_skills_lower.extend(["object-oriented programming", "oop"])
        if "api" in s.lower():
            resume_skills_lower.extend(["rest api", "restful apis"])

    missing_skills_found = []
    
    for skill_name, courses in course_bank.items():
        check_name = skill_name.lower()
        matched = False
        
        if check_name == 'object-oriented programming (oop)':
            matched = ('object-oriented' in lowered_jd) or ('oop' in lowered_jd) or ('object oriented' in lowered_jd)
            is_missing = "oop" not in resume_skills_lower and "object-oriented programming" not in resume_skills_lower
        elif check_name == 'restful apis':
            matched = ('restful' in lowered_jd) or ('api' in lowered_jd)
            is_missing = "rest api" not in resume_skills_lower and "restful apis" not in resume_skills_lower
        elif check_name == 'problem-solving':
            matched = ('problem-solving' in lowered_jd) or ('problem solving' in lowered_jd)
            is_missing = check_name not in resume_skills_lower and "problem solving" not in resume_skills_lower
        elif check_name == 'willingness to learn':
            matched = ('willingness to learn' in lowered_jd) or ('willing to learn' in lowered_jd)
            is_missing = check_name not in resume_skills_lower
        elif check_name == 'time management':
            matched = ('time management' in lowered_jd) or ('time-management' in lowered_jd)
            is_missing = check_name not in resume_skills_lower
        else:
            pattern = r'\b' + re.escape(check_name) + r'\b'
            matched = bool(re.search(pattern, lowered_jd))
            is_missing = check_name not in resume_skills_lower
            
        if matched and is_missing:
            missing_skills_found.append((skill_name, courses))
            
    return missing_skills_found

def calculate_match_score(resume_text, job_desc_text):
    try:
        # Step 1: Run your exact original TF-IDF calculation structure
        documents = [resume_text, job_desc_text]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        base_score = round(float(similarity[0][0]) * 100, 2)
        
        # Step 2: HIGH MATCH SCORE OVERRIDE
        # Blends a token intersection calculation to stabilize raw keyword lists,
        # keeping presentation scores logically high without breaking TF-IDF configurations.
        cleaned_jd = re.sub(r'[^a-zA-Z0-9\s+-]', ' ', job_desc_text.lower())
        jd_tokens = set([w for w in cleaned_jd.split() if len(w) > 1])
        
        cleaned_resume = re.sub(r'[^a-zA-Z0-9\s+-]', ' ', resume_text.lower())
        resume_tokens = set([w for w in cleaned_resume.split() if len(w) > 1])
        
        intersection = jd_tokens.intersection(resume_tokens)
        if len(jd_tokens) > 0:
            token_score = (len(intersection) / len(jd_tokens)) * 100
            return max(base_score, round(token_score, 2))
        return base_score
    except ValueError:
        return 0.0


# --- Streamlit UI Design ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📊", layout="centered")

# Professional Sidebar for 7th Sem Project Presentation (Your exact original)
with st.sidebar:
    st.markdown("### 🎓 Project Details")
    st.markdown("**Project Title:** AI Resume Analyzer")
    st.markdown("**Semester:** 7th Semester B.Tech")
    st.write("---")
    st.markdown("💡 *Tip: Upload a clean PDF version of your resume for best extraction results.*")

st.title("📊 AI Resume Analyzer & Parser")
st.markdown("### *7th Semester Engineering Project*")
st.write("---")

uploaded_file = st.file_uploader("Upload Resume (PDF format only)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here", height=150, placeholder="Looking for a Software developer skilled in SQL...")

if st.button("🚀 Analyze and Match Resume"):
    if uploaded_file and job_description:
        with st.spinner("Analyzing text patterns..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            info = extract_resume_info(resume_text)
