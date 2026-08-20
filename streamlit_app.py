import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Core Skill Bank and Course Mapping ---
SKILL_DATABASE = {
    # 1. Core Technical Languages & Frameworks
    "python": "Python Basics for Data Science & Software (Coursera)",
    "java": "Java Programming: Solving Problems with Software (Coursera)",
    "c++": "C++ Programming For Beginners (Udemy)",
    "c": "C Programming Language Fundamentals (Pluralsight)",
    "javascript": "JavaScript Basics & DOM Manipulation (freeCodeCamp)",
    "html": "Responsive Web Design Certification (freeCodeCamp)",
    "css": "CSS Basics and Flexbox (freeCodeCamp / Scrimba)",
    "bootstrap": "Bootstrap 5 Responsive Web Design (Udemy)",
    "sql": "Intro to SQL: Querying and Managing Data (Khan Academy / Udemy)",
    "mysql": "MySQL Database Development (Coursera)",
    "rest api": "REST API Design, Development & Management (Udemy)",
    "api": "REST API Design, Development & Management (Udemy)",
    "git": "Version Control with Git and GitHub for Beginners (Udemy)",
    "github": "Git & GitHub Crash Course (freeCodeCamp)",
    # 2. Engineering Methodologies & CS Core
    "data structures": "Data Structures & Algorithms (GeeksforGeeks)",
    "algorithms": "Algorithms Specialization for Beginners (Coursera)",
    "oop": "Object Oriented Programming Fundamentals (Udemy)",
    "object oriented programming": "Object Oriented Programming Fundamentals (Udemy)",
    "problem solving": "Problem Solving Fundamentals (HackerRank)",
    "agile": "Agile Software Development & Frameworks (Coursera)",
    "scrum": "Scrum Master & Sprint Planning Fundamentals (Udemy)",
    # 3. Essential Soft Skills
    "communication": "Communication Skills for Engineers (Coursera)",
    "teamwork": "Teamwork & Collaboration in Tech (LinkedIn Learning)",
    "collaboration": "Effective Collaboration in Agile Teams (Coursera)",
    "adaptability": "Adaptability & Agility in Software Development (Udemy)",
    "time management": "Time Management for Software Developers (Pluralsight)",
    "critical thinking": "Critical Thinking & Debugging Mindset (LinkedIn Learning)",
}

SYNONYMS = {
    "oop": "object oriented programming",
    "object oriented programming": "oop",
    "git": "github",
    "github": "git",
    "sql": "mysql",
    "mysql": "sql",
    "api": "rest api",
    "rest api": "api",
    "teamwork": "collaboration",
    "collaboration": "teamwork",
    "agile": "scrum",
    "scrum": "agile",
}

# Skill Categorization Keys
METHODOLOGY_SKILLS_KEYS = [
    "data structures",
    "algorithms",
    "oop",
    "object oriented programming",
    "problem solving",
    "agile",
    "scrum",
]

SOFT_SKILLS_KEYS = [
    "communication",
    "teamwork",
    "collaboration",
    "adaptability",
    "time management",
    "critical thinking",
]


def extract_raw_text_from_pdf(pdf_file):
    """Safely extracts raw text preserving line breaks for contact header parsing."""
    try:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        text_pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_pages.append(t)
        return "\n".join(text_pages)
    except Exception:
        return ""


def extract_name(raw_text):
    """Extracts candidate name by inspecting line-by-line structure."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    if not lines:
        return "Not Found"

    for line in lines[:5]:
        if re.search(
            r"\b(resume|curriculum|vitae|summary|experience|education|skills)\b",
            line,
            re.I,
        ):
            continue

        cleaned_line = re.sub(email_pattern, "", line)
        cleaned_line = re.sub(
            r"https?://\S+|www\.\S+|\+?\d[\d\s.-]{8,}", "", cleaned_line
        )
        cleaned_line = re.sub(r"[|•·–-]", " ", cleaned_line)
        cleaned_line = re.sub(r"[^a-zA-Z\s]", "", cleaned_line)
        cleaned_line = re.sub(r"\s+", " ", cleaned_line).strip()

        words = cleaned_line.split()
        if 2 <= len(words) <= 4:
            return " ".join(words).title()

    return "Not Found"


def extract_skills_from_text(text):
    """Extracts database skills present in text using boundary patterns."""
    text_clean = re.sub(r"[^\w\s+]", " ", text.lower())
    extracted = []

    for skill in SKILL_DATABASE.keys():
        pattern = (
            r"(?i)(?<![a-zA-Z0-9#+])" + re.escape(skill) + r"(?![a-zA-Z0-9#+])"
        )
        if re.search(pattern, text_clean):
            extracted.append(skill.lower())

    return list(set(extracted))


def extract_resume_info(raw_text):
    name = extract_name(raw_text)
    clean_single_line = re.sub(r"\s+", " ", raw_text)

    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email = re.findall(email_pattern, raw_text)

    phone_pattern = r"(?:\+?\d{1,3}[\s.-]*)?(?:\(?\d{2,5}\)?[\s.-]*)?\d{3,5}[\s.-]*\d{3,5}"
    phone_matches = re.findall(phone_pattern, clean_single_line)

    phone = "Not Found"
    for match in phone_matches:
        digits_only = re.sub(r"\D", "", match)
        if 10 <= len(digits_only) <= 13:
            phone = match.strip()
            break

    extracted_skills = extract_skills_from_text(raw_text)

    return {
        "Name": name,
        "Email": email[0] if email else "Not Found",
        "Phone": phone,
        "Skills": extracted_skills,
        "NormalizedText": clean_single_line,
    }


def calculate_match_score(resume_text, job_desc_text, resume_skills):
    job_desc_clean = re.sub(r"[^\w\s+]", " ", job_desc_text.lower())
    resume_text_clean = re.sub(r"[^\w\s+]", " ", resume_text.lower())

    # 1. Normalize resume skills using synonyms
    normalized_resume_skills = set()
    for s in resume_skills:
        s_lower = s.lower().strip()
        canonical_s = SYNONYMS.get(s_lower, s_lower)
        normalized_resume_skills.add(canonical_s)

    # 2. Extract skills explicitly required in Job Description
    jd_skills = extract_skills_from_text(job_desc_text)

    canonical_required = set()
    for req in jd_skills:
        canonical = SYNONYMS.get(req, req)
        canonical_required.add(canonical)

    # 3. Calculate Skill Match Percentage
    if canonical_required:
        matched_count = sum(
            1 for req in canonical_required if req in normalized_resume_skills
        )
        skill_score = (matched_count / len(canonical_required)) * 100.0
    else:
        if normalized_resume_skills:
            matched_count = sum(
                1 for s in normalized_resume_skills if s in job_desc_clean
            )
            skill_score = (
                matched_count / len(normalized_resume_skills)
            ) * 100.0
        else:
            skill_score = 0.0

    # 4. Calculate Text Similarity (TF-IDF)
    try:
        documents = [resume_text_clean, job_desc_clean]
        vectorizer = TfidfVectorizer(
            stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        vector_sim = float(
            cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        )
        tfidf_score = min(100.0, vector_sim * 250.0)
    except Exception:
        tfidf_score = skill_score

    # 5. Final Score Calculation
    if skill_score >= 80.0:
        final_score = max(90.0, (skill_score * 0.85) + (tfidf_score * 0.15))
    elif skill_score > 0:
        final_score = (skill_score * 0.80) + (tfidf_score * 0.20)
    else:
        final_score = tfidf_score * 0.50

    return min(round(final_score, 2), 100.0)


def get_course_recommendations(resume_skills, job_desc_text):
    job_text_clean = re.sub(r"[^\w\s+]", " ", job_desc_text.lower())

    normalized_resume_skills = set()
    for s in resume_skills:
        s_lower = s.lower().strip()
        canonical_s = SYNONYMS.get(s_lower, s_lower)
        normalized_resume_skills.add(canonical_s)

    missing_tech, tech_recs = [], []
    missing_method, method_recs = [], []
    missing_soft, soft_recs = [], []

    for skill, course in SKILL_DATABASE.items():
        pattern = (
            r"(?i)(?<![a-zA-Z0-9#+])" + re.escape(skill) + r"(?![a-zA-Z0-9#+])"
        )
        if re.search(pattern, job_text_clean):
            canonical_skill = SYNONYMS.get(skill, skill)
            if canonical_skill not in normalized_resume_skills:
                if skill in SOFT_SKILLS_KEYS:
                    if course not in soft_recs:
                        missing_soft.append(skill.title())
                        soft_recs.append(course)
                elif skill in METHODOLOGY_SKILLS_KEYS:
                    if course not in method_recs:
                        missing_method.append(skill.title())
                        method_recs.append(course)
                else:
                    if course not in tech_recs:
                        missing_tech.append(skill.title())
                        tech_recs.append(course)

    # Fallbacks for empty categories
    if not missing_soft:
        for soft_key in ["communication", "teamwork"]:
            canonical_soft = SYNONYMS.get(soft_key, soft_key)
            if canonical_soft not in normalized_resume_skills:
                course = SKILL_DATABASE[soft_key]
                if course not in soft_recs:
                    missing_soft.append(soft_key.title())
                    soft_recs.append(course)

    return missing_tech, tech_recs, missing_method, method_recs, missing_soft, soft_recs


# --- Streamlit UI Design ---
st.set_page_config(
    page_title="AI Resume Analyzer", page_icon="📊", layout="centered"
)

with st.sidebar:
    st.markdown("### 🎓 Project Details")
    st.markdown("**Project Title:** AI Resume Analyzer & Parser")
    st.markdown("**Semester:** 7th Semester, B.Tech")
    st.write("---")
    st.markdown(
        "💡 *Tip: Upload a clean PDF version of your resume for best extraction results.*"
    )

st.title("📊 AI Resume Analyzer & Parser")
st.markdown("### *7th Semester Engineering Project*")
st.write("---")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF format only)", type=["pdf"]
)
job_description = st.text_area(
    "Paste Job Description Here",
    height=150,
    placeholder="Looking for a Junior Software Developer proficient in Java, SQL, Git, Agile/Scrum methodologies, and team collaboration...",
)

if st.button("🚀 Analyze and Match Resume"):
    has_pdf = uploaded_file is not None
    has_jd = bool(job_description.strip())

    if not has_pdf and not has_jd:
        st.error(
            "⚠️ Please upload your Resume PDF and paste the Job Description text to begin analysis."
        )
    elif not has_pdf and has_jd:
        st.warning(
            "⚠️ Job Description detected! Please upload your Resume PDF to complete the matching process."
        )
    elif has_pdf and not has_jd:
        st.warning(
            "⚠️ Resume uploaded! Please paste the Job Description text below to calculate your match score."
        )
    else:
        with st.spinner("Analyzing text patterns..."):
            try:
                raw_resume_text = extract_raw_text_from_pdf(uploaded_file)

                if not raw_resume_text.strip():
                    st.error(
                        "⚠️ Could not extract text from this PDF. Please ensure it is a text-based PDF file."
                    )
                else:
                    info = extract_resume_info(raw_resume_text)
                    match_score = calculate_match_score(
                        info["NormalizedText"], job_description, info["Skills"]
                    )

                    st.metric(label="🎯 Job Match Score", value=f"{match_score}%")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📧 Contact Details")
                        st.markdown(f"**Name:** {info['Name']}")
                        st.markdown(f"**Email:** {info['Email']}")
                        st.markdown(f"**Phone:** {info['Phone']}")

                    with col2:
                        st.subheader("🛠️ Extracted Skills")
                        if info["Skills"]:
                            display_skills = [s.title() for s in info["Skills"]]
                            st.success(", ".join(display_skills))
                        else:
                            st.warning("No developer skills detected.")

                    st.write("---")

                    st.subheader("💡 Recommended Skill Upgrades")
                    (
                        m_tech, rec_tech,
                        m_method, rec_method,
                        m_soft, rec_soft
                    ) = get_course_recommendations(info["Skills"], job_description)

                    # 1. Tech Skills UI
                    if rec_tech:
                        st.markdown(f"**🛠️ Missing Technical Skills:** {', '.join(m_tech)}")
                        for course in rec_tech:
                            st.write(f"- {course}")
                    else:
                        st.success("✅ Technical skills match all JD requirements!")

                    st.write("---")

                    # 2. Methodologies UI
                    if rec_method:
                        st.markdown(f"**⚙️ Methodologies & CS Core to Focus On:** {', '.join(m_method)}")
                        for course in rec_method:
                            st.write(f"- {course}")
                    else:
                        st.success("✅ Engineering methodologies match expectations!")

                    st.write("---")

                    # 3. Soft Skills UI
                    if rec_soft:
                        st.markdown(f"**🤝 Essential Soft Skills to Highlight:** {', '.join(m_soft)}")
                        for course in rec_soft:
                            st.write(f"- {course}")
                    else:
                        st.success("✅ Soft skill profile matches role expectations!")

            except Exception as e:
                st.error(f"⚠️ Error parsing PDF file: {str(e)}")
