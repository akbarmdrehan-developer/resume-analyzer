import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Core Skill Bank and Course Mapping ---
SKILL_DATABASE = {
    # Core Languages
    "python": "Python Basics for Data Science & Software (Coursera)",
    "java": "Java Programming: Solving Problems with Software (Coursera)",
    "c++": "C++ Programming For Beginners (Udemy)",
    "c": "C Programming Language Fundamentals (Pluralsight)",
    "javascript": "JavaScript Basics & DOM Manipulation (freeCodeCamp)",
    # Web & UI Frameworks
    "html": "Responsive Web Design Certification (freeCodeCamp)",
    "css": "CSS Basics and Flexbox (freeCodeCamp / Scrimba)",
    "bootstrap": "Bootstrap 5 Responsive Web Design (Udemy)",
    # Databases & Backend
    "sql": "Intro to SQL: Querying and Managing Data (Khan Academy / Udemy)",
    "mysql": "MySQL Database Development (Coursera)",
    "rest api": "REST API Design, Development & Management (Udemy)",
    # Computer Science & Tools
    "data structures": "Data Structures & Algorithms (GeeksforGeeks)",
    "algorithms": "Algorithms Specialization for Beginners (Coursera)",
    "git": "Version Control with Git and GitHub for Beginners (Udemy)",
    "github": "Git & GitHub Crash Course (freeCodeCamp)",
    "oop": "Object Oriented Programming Fundamentals (Udemy)",
    "object oriented programming": "Object Oriented Programming Fundamentals (Udemy)",
    "problem solving": "Problem Solving Fundamentals (HackerRank)",
}

SYNONYMS = {
    "oop": "object oriented programming",
    "object oriented programming": "oop",
    "git": "github",
    "github": "git",
    "sql": "mysql",
    "mysql": "sql",
}


def extract_raw_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_name(raw_text):
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


def extract_resume_info(raw_text):
    name = extract_name(raw_text)
    normalized_text = re.sub(r"\s+", " ", raw_text)

    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email = re.findall(email_pattern, normalized_text)

    phone_pattern = r"(?:\+?\d{1,3}[\s.-]*)?(?:\(?\d{2,5}\)?[\s.-]*)?\d{3,5}[\s.-]*\d{3,5}"
    phone_matches = re.findall(phone_pattern, normalized_text)

    phone = "Not Found"
    for match in phone_matches:
        digits_only = re.sub(r"\D", "", match)
        if 10 <= len(digits_only) <= 13:
            phone = match.strip()
            break

    lowered_text = normalized_text.lower()
    extracted_skills = []

    for skill in SKILL_DATABASE.keys():
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lowered_text):
            extracted_skills.append(skill.lower())

    return {
        "Name": name,
        "Email": email[0] if email else "Not Found",
        "Phone": phone,
        "Skills": list(set(extracted_skills)),
        "NormalizedText": normalized_text,
    }


def calculate_match_score(resume_text, job_desc_text, resume_skills):
    job_desc_lower = job_desc_text.lower()

    # Identify database skills present in the Job Description
    required_skills = [
        s
        for s in SKILL_DATABASE.keys()
        if re.search(r"\b" + re.escape(s) + r"\b", job_desc_lower)
    ]

    resume_skills_lower = [s.lower() for s in resume_skills]

    if required_skills:
        matched_count = 0
        for req in required_skills:
            if (
                req in resume_skills_lower
                or SYNONYMS.get(req) in resume_skills_lower
            ):
                matched_count += 1

        skill_score = (matched_count / len(required_skills)) * 100
    else:
        # Fallback: calculate skill score based on how many extracted resume skills are in JD
        if resume_skills_lower:
            found_in_jd = sum(
                1 for s in resume_skills_lower if s in job_desc_lower
            )
            skill_score = (found_in_jd / len(resume_skills_lower)) * 100
        else:
            skill_score = 0.0

    # TF-IDF Vector Similarity Calculation
    try:
        documents = [resume_text, job_desc_text]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        vector_sim = float(
            cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        )
        tfidf_score = vector_sim * 100
    except ValueError:
        tfidf_score = 0.0

    final_score = (skill_score * 0.70) + (tfidf_score * 0.30)
    return min(round(final_score, 2), 100.0)


def get_course_recommendations(resume_skills, job_desc_text):
    job_text_lower = job_desc_text.lower()
    resume_skills_lower = [s.lower() for s in resume_skills]

    missing_skills = []
    recommendations = []

    for skill, course in SKILL_DATABASE.items():
        if re.search(r"\b" + re.escape(skill) + r"\b", job_text_lower):
            if (
                skill not in resume_skills_lower
                and SYNONYMS.get(skill) not in resume_skills_lower
            ):
                if course not in recommendations:
                    missing_skills.append(skill.title())
                    recommendations.append(course)

    return missing_skills, recommendations


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
    placeholder="Looking for a Junior Software Developer proficient in Java, SQL, Git, and Data Structures...",
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
            raw_resume_text = extract_raw_text_from_pdf(uploaded_file)
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
                    st.warning("No beginner developer skills detected.")

            st.write("---")

            st.subheader("💡 Skill Upgrade & Course Recommendations")
            missing_skills, rec_courses = get_course_recommendations(
                info["Skills"], job_description
            )

            # Strict recommendation logic fix
            if rec_courses:
                st.info(
                    f"Missing Skill Areas to Focus On: **{', '.join(missing_skills)}**"
                )
                for i, course in enumerate(rec_courses, 1):
                    st.write(f"**{i}.** {course}")
            elif match_score >= 80.0:
                st.success(
                    "🎉 Exceptional match! Your skill profile covers all core requirements detected in this job description."
                )
            else:
                st.warning(
                    "⚠️ Low score match. Try adding more relevant technical skills, project details, and keywords from the job description to your resume."
)
