import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def extract_name(text):
    # Standard email/phone/url removal to avoid mistaking them for names
    cleaned_first_lines = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines[:5]:  # Look closely at the top 5 lines
        if not re.search(
            r"@|http|www|\d|\b(resume|curriculum|vitae)\b", line, re.I
        ):
            # Check if line looks like a valid name (2-4 capitalized words)
            if re.match(
                r"^[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){1,3}$", line
            ):
                return line.title()
    return "Not Found"


def extract_resume_info(text):
    # Extract Name
    name = extract_name(text)

    # Extract Email using Regex
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email = re.findall(email_pattern, text)

    # Extract Phone Numbers
    phone_pattern = (
        r"\b(?:\+?\d{1,3}[-. \s]?)?\(?\d{3}\)?[-. \s]?\d{3}[-. \s]?\d{4}\b"
    )
    phone = re.findall(phone_pattern, text)

    # Tailored Software Developer Skill Bank
    skill_bank = [
        "python",
        "java",
        "c++",
        "c#",
        "javascript",
        "typescript",
        "sql",
        "nosql",
        "react",
        "angular",
        "vue.js",
        "node.js",
        "express",
        "django",
        "flask",
        "fastapi",
        "spring boot",
        "html",
        "css",
        "bootstrap",
        "tailwind",
        "git",
        "github",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "ci/cd",
        "rest api",
        "graphql",
        "microservices",
        "data structures",
        "algorithms",
        "system design",
        "linux",
        "postgresql",
        "mongodb",
        "redis",
    ]

    lowered_text = text.lower()
    extracted_skills = []

    for skill in skill_bank:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lowered_text):
            extracted_skills.append(skill.title())

    return {
        "Name": name,
        "Email": email[0] if email else "Not Found",
        "Phone": phone[0] if phone else "Not Found",
        "Skills": list(set(extracted_skills)),
    }


def calculate_match_score(resume_text, job_desc_text):
    try:
        documents = [resume_text, job_desc_text]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return round(float(similarity[0][0]) * 100, 2)
    except ValueError:
        return 0.0


def get_course_recommendations(resume_skills, job_desc_text):
    # Mapping high-demand skills to recommended learning resources
    course_database = {
        "Python": "Python for Everybody Specialization (Coursera / Univ. of Michigan)",
        "Java": "Java Programming and Software Engineering Fundamentals (Coursera)",
        "React": "React - The Complete Guide (Udemy)",
        "Node.Js": "The Complete Node.js Developer Course (Udemy)",
        "Docker": "Docker & Kubernetes: The Practical Guide (Udemy)",
        "Aws": "AWS Certified Developer Associate (A Cloud Guru / Udemy)",
        "Sql": "The Complete SQL Bootcamp (Udemy)",
        "Spring Boot": "Spring Boot Fundamentals (Pluralsight / Udemy)",
        "System Design": "Grokking the System Design Interview (Educative.io)",
        "Data Structures": "Data Structures and Algorithms Specialization (Coursera)",
        "Git": "Git Complete: The definitive step-by-step guide (Udemy)",
        "Kubernetes": "Certified Kubernetes Administrator (CKA) (Linux Foundation)",
    }

    job_text_lower = job_desc_text.lower()
    resume_skills_lower = [s.lower() for s in resume_skills]

    missing_skills = []
    recommendations = []

    for skill, course in course_database.items():
        # Skill is requested in Job Description but missing from candidate's resume
        if (
            skill.lower() in job_text_lower
            and skill.lower() not in resume_skills_lower
        ):
            missing_skills.append(skill)
            recommendations.append(course)

    return missing_skills, recommendations


# --- Streamlit UI Design ---
st.set_page_config(
    page_title="AI Resume Analyzer", page_icon="📊", layout="centered"
)

# Sidebar
with st.sidebar:
    st.markdown("### 🎓 Project Details")
    st.markdown("**Project Title:** AI Resume Analyzer")
    st.markdown("**Semester:** 7th Semester B.Tech")
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
    placeholder="Looking for a Software Developer skilled in React, Node.js, AWS, and SQL...",
)

if st.button("🚀 Analyze and Match Resume"):
    if uploaded_file and job_description:
        with st.spinner("Analyzing text patterns..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            info = extract_resume_info(resume_text)
            match_score = calculate_match_score(resume_text, job_description)

            # Match Score Card
            st.metric(label="🎯 Job Match Score", value=f"{match_score}%")

            # Display Extracted Data
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📧 Contact Details")
                st.markdown(f"**Name:** {info['Name']}")
                st.markdown(f"**Email:** {info['Email']}")
                st.markdown(f"**Phone:** {info['Phone']}")

            with col2:
                st.subheader("🛠️ Extracted Skills")
                if info["Skills"]:
                    st.success(", ".join(info["Skills"]))
                else:
                    st.warning("No matching developer skills detected.")

            st.write("---")

            # Course Recommendations Section
            st.subheader("💡 Course Recommendations to Boost Match Score")
            missing_skills, rec_courses = get_course_recommendations(
                info["Skills"], job_description
            )

            if rec_courses:
                st.info(
                    f"We found missing skills required in the job description: **{', '.join(missing_skills)}**"
                )
                for i, course in enumerate(rec_courses, 1):
                    st.write(f"**{i}.** {course}")
            else:
                st.success(
                    "🎉 Great job! Your technical skill profile matches the core requirements of this job description."
                )

    else:
        st.error(
            "⚠️ Please provide both the resume PDF and the job description text."
        )
