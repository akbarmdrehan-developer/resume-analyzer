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
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Not Found"
    
    corrupt_words = {'resume', 'cv', 'curriculum', 'vitae', 'page', 'summary', 'profile'}
    
    for line in lines[:3]:
        if (len(line) < 30 and 
            not any(char.isdigit() for char in line) and 
            '@' not in line and 
            line.lower() not in corrupt_words):
            name = re.sub(r'\s+', ' ', line)
            return name
            
    return "Not Found"

def extract_resume_info(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_matches = re.findall(email_pattern, text)
    email = email_matches[0].strip() if email_matches else "Not Found"
    
    phone_pattern = r'(?:\+?\d{1,3}[-. \s]?)?\(?\d{3}\)?[-. \s]?\d{3}[-. \s]?\d{4}\b'
    phone_matches = re.findall(phone_pattern, text)
    phone = phone_matches[0].strip() if phone_matches else "Not Found"
    
    skill_bank = [
        'java', 'python', 'c++', 'javascript',
        'data structures', 'algorithms', 'object-oriented programming', 'oop',
        'sql', 'mysql', 'mongodb',
        'html', 'css', 'restful apis', 'rest api',
        'git', 'github',
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
        "Email": email, 
        "Phone": phone, 
        "Skills": list(set(extracted_skills))
    }

def recommend_courses(resume_skills, job_desc_text):
    course_bank = {
        'Java': ['Java Programming and Software Engineering Fundamentals (Coursera)', 'Java Masterclass (Udemy)'],
        'Python': ['Python for Everybody Specialization (Coursera)', 'Complete Python Bootcamp (Udemy)'],
        'C++': ['Coding in C++ (edX)', 'Beginning C++ Programming (Udemy)'],
        'Javascript': ['JavaScript: The Advanced Concepts (Udemy)', 'Modern JavaScript From The Beginning (Udemy)'],
        'Data Structures': ['Data Structures and Algorithms Specialization (Coursera)', 'Master the Coding Interview: DSA (Udemy)'],
        'Algorithms': ['Algorithms Specialization by Stanford (Coursera)', 'Introduction to Algorithms (MIT OpenCourseWare)'],
        'Object-Oriented Programming (OOP)': ['Object Oriented Programming in Java/C++ (Udemy)', 'OOP Design Patterns (Coursera)'],
        'SQL': ['SQL for Data Science (Coursera)', 'The Complete SQL Bootcamp (Udemy)'],
        'Mysql': ['The Ultimate MySQL Bootcamp (Udemy)', 'MySQL Database Administration track'],
        'Mongodb': ['MongoDB - The Complete Developer\'s Guide (Udemy)', 'MongoDB University Free Courses'],
        'HTML': ['Introduction to HTML5 (Coursera)', 'Web Design for Beginners (Udemy)'],
        'CSS': ['Advanced CSS and Sass (Udemy)', 'CSS - The Complete Guide (Udemy)'],
        'RESTful APIs': ['API Design and Fundamentals (Google/Coursera)', 'REST API Design (Udemy)'],
        'Git': ['Version Control with Git (Coursera)', 'Git & GitHub Masterclass (Udemy)'],
        'Github': ['GitHub Ultimate: Master Git and GitHub (Udemy)', 'Introduction to GitHub (GitHub Skills Track)'],
        'Problem-Solving': ['Creative Problem Solving & Decision Making (Coursera)', 'Effective Problem-Solving Frameworks (LinkedIn Learning)'],
        'Analytical Thinking': ['Critical Thinking and Problem Solving (edX)', 'Introduction to Analytical Thinking (Coursera)'],
        'Communication': ['Improving Communication Skills (Coursera)', 'Effective Professional Communication (edX)'],
        'Teamwork': ['Teamwork Skills & Collaborating Effectively (Coursera)'],
        'Collaboration': ['High-Performance Collaboration: Leadership, Teamwork, and Negotiation (Coursera)'],
        'Adaptability': ['Developing Adaptability & Resilience in the Workplace (LinkedIn Learning)'],
        'Willingness To Learn': ['Learning How to Learn by Barbara Oakley (Coursera)'],
        'Time Management': ['Work Smarter, Not Harder: Time Management for Personal & Professional Productivity (Coursera)']
    }
    
    lowered_jd = job_desc_text.lower()
    
    resume_skills_lower = []
    for s in resume_skills:
        resume_skills_lower.append(s.lower())
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
            matched = check_name in lowered_jd
            is_missing = check_name not in resume_skills_lower
            
        if matched and is_missing:
            missing_skills_found.append((skill_name, courses))
            
    return missing_skills_found

def calculate_match_score(resume_text, job_desc_text):
    try:
        documents = [resume_text, job_desc_text]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(float(similarity[0][0]) * 100, 2)
    except ValueError:
        return 0.0


# --- Streamlit UI Design ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📊", layout="centered")

with st.sidebar:
    st.markdown("### 🎓 Project Details")
    st.markdown("**Project Title:** AI Resume Analyzer")
    st.markdown("**Semester:** 7th Semester B.E./B.Tech")
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
            match_score = calculate_match_score(resume_text, job_description)
            
            st.metric(label="🎯 Job Match Score", value=f"{match_score}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("👤 Candidate Profile")
                st.markdown(f"**Name:** {info['Name']}")
                st.markdown(f"**Email:** {info['Email']}")
                st.markdown(f"**Phone:** {info['Phone']}")
            with col2:
                st.subheader("🛠️ Extracted Skills")
                if info['Skills']:
                    st.success(", ".join(info['Skills']))
                else:
                    st.warning("No standard technical skills detected.")
            
            st.write("---")
            st.subheader("📚 Upskilling & Course Recommendations")
            st.markdown("Based on the target Job Description, acquire these missing skills to maximize your score:")
            
            recommended_courses_list = recommend_courses(info['Skills'], job_description)
