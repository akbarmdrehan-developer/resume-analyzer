import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += "\n".join([line.strip() for line in page_text.split('\n') if line.strip()]) + "\n"
    return text

def extract_name(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Not Found"
    
    corrupt_words = {'resume', 'cv', 'curriculum', 'vitae', 'page', 'summary', 'profile'}
    
    for line in lines[:4]:
        line_clean = re.sub(r'\s+', ' ', line).strip()
        if (len(line_clean) < 30 and 
            not any(char.isdigit() for char in line_clean) and 
            '@' not in line_clean and 
            line_clean.lower() not in corrupt_words):
            return line_clean
            
    return "Not Found"

def extract_resume_info(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_matches = re.findall(email_pattern, text)
    # FIXED: Extract string index 0 safely before attempting strip operation
    email = email_matches[0].strip() if email_matches else "Not Found"
    
    phone_pattern = r'(?:\+?\d{1,3}[-. \s]?)?\(?\d{3}\)?[-. \s]?\d{3}[-. \s]?\d{4}\b'
    phone_matches = re.findall(phone_pattern, text)
    # FIXED: Extract string index 0 safely before attempting strip operation
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
    
    lowered_text = " " + " ".join(text.lower().split()) + " "
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
    
    lowered_jd = " " + " ".join(job_desc_text.lower().split()) + " "
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    for s in resume_skills:
        if "object-oriented" in s.lower() or "oop" in s.lower():
            resume_skills_lower.extend(["object-oriented programming", "oop", "object-oriented programming (oop)"])
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

def calculate_match_score(resume_skills, job_desc_text):
    lowered_jd = " " + " ".join(job_desc_text.lower().split()) + " "
    target_skills_in_jd = 0
    matched_skills = 0
    
    skill_check_list = [
        ('java', ['java']), ('python', ['python']), ('c++', ['c++']), ('javascript', ['javascript']),
        ('data structures', ['data structures']), ('algorithms', ['algorithms']),
        ('object-oriented programming (oop)', ['object-oriented', 'oop', 'object oriented']),
        ('sql', ['sql']), ('mysql', ['mysql']), ('mongodb', ['mongodb']),
        ('html', ['html']), ('css', ['css']), ('restful apis', ['restful', 'api', 'apis']),
        ('git', ['git']), ('github', ['github']), ('problem-solving', ['problem-solving', 'problem solving']),
        ('analytical thinking', ['analytical thinking', 'analytical']), ('communication', ['communication']),
        ('teamwork', ['teamwork']), ('collaboration', ['collaboration']), ('adaptability', ['adaptability']),
        ('willingness to learn', ['willingness to learn', 'willing to learn']), ('time management', ['time management'])
    ]
    
    resume_skills_lower = [s.lower() for s in resume_skills]
    
    for formal_name, triggers in skill_check_list:
        jd_demands_skill = any(t in lowered_jd for t in triggers)
        if jd_demands_skill:
            target_skills_in_jd += 1
            has_skill = any(t in resume_skills_lower or formal_name in resume_skills_lower for t in triggers)
            if has_skill:
                matched_skills += 1
                
    if target_skills_in_jd == 0:
        return 100.0
        
    return round((matched_skills / target_skills_in_jd) * 100, 2)


# --- Streamlit UI Design ---
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📊", layout="centered")

with st.sidebar:
    st.markdown("### 🎓 Project Details")
    st.markdown("**Project Title:** AI Resume Analyzer")
    st.markdown("**Semester:** 7th Semester B.E./B.Tech")
    st.write("---")

