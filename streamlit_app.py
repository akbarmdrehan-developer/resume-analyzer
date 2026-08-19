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

    # Normalize unicode whitespace/non-breaking spaces to standard spaces
    normalized_text = re.sub(r"\s+", " ", text)
    return normalized_text


def extract_name(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return "Not Found"

    for line in lines[:3]:
        cleaned_line = re.sub(email_pattern, "", line)
        cleaned_line = re.sub(
            r"https?://\S+|www\.\S+|\+?\d[\d\s.-]{8,}", "", cleaned_line
        )
        cleaned_line = re.sub(r"[|•·–-]", " ", cleaned_line)
        cleaned_line = re.sub(r"\s+", " ", cleaned_line).strip()

        words = cleaned_line.split()
        if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
            return " ".join(words).title()

    return "Not Found"


def extract_resume_info(text):
    name = extract_name(text)

    # Email extraction
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email = re.findall(email_pattern, text)

    # Phone extraction
    phone_pattern = r"(?:\+?\d{1,3}[\s.-]*)?(?:\(?\d{2,5}\)?[\s.-]*)?\d{3,5}[\s.-]*\d{3,5}"
    phone_matches = re.findall(phone_pattern, text)

    phone = "Not Found"
    for match in phone_matches:
        digits_only = re.sub(r"\D", "", match)
        if 10 <= len(digits_only) <= 13:
            phone = match.strip()
            break

    # Beginner Software Developer Skill Bank
    skill_bank = [
        "python",
        "java",
        "c++",
        "c",
        "javascript",
        "html",
        "css",
        "bootstrap",
        "sql",
        "mysql",
        "git",
        "github",
        "data structures",
        "algorithms",
        "oop",
        "object oriented programming",
        "rest api",
        "problem solving",
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
        "Phone": phone,
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


def get_course_recommendations(resume_skills, job_desc_text, match_score):
    # Beginner-focused course database
    course_database = {
        "Python": "Python Basics for Data Science & Software (Coursera)",
        "Java": "Java Programming: Solving Problems with Software (Coursera)",
        "Javascript": "JavaScript Basics & DOM Manipulation (freeCodeCamp)",
        "Sql": "Intro to SQL: Querying and Managing Data (Khan Academy / Udemy)",
        "Git": "Version Control with Git and GitHub for Beginners (Udemy)",
        "Data Structures": "Data Structures & Algorithms in Python/Java (GeeksforGeeks / Coursera)",
        "Algorithms": "Algorithms Specialization for Beginners (Coursera)",
        "Html": "Responsive Web Design Certification (freeCodeCamp)",
        "Css": "CSS Basics and Flexbox (freeCodeCamp / Scrimba)",
        "Oop": "Object Oriented Programming Fundamentals (Udemy)",
    }

    job_text_lower = job_desc_text.lower()
    resume_skills_lower = [s.lower() for s in resume_skills]

    missing_skills = []
    recommendations = []

    for skill, course in course_database.items():
        if (
            skill.lower() in job_text_lower
            and skill.lower() not in resume_skills_lower
        ):
            missing_skills.append(skill)
            recommendations.append(course)

    # Entry-level fallback recommendations if match score is low
    if match_score < 50.0 and not recommendations:
        missing_skills.append("Software Development Fundamentals")
        recommendations.extend(
            [
                "CS50's Introduction to Computer Science (Harvard / edX)",
                "Data Structures and Algorithms for Beginners (freeCodeCamp)",
                "Git & GitHub Starter Crash Course (Udemy)",
            ]
        )

    return missing_skills, recommendations


# --- Streamlit UI Design ---
st.set_page_config(
    page_title="AI Resume Analyzer", page_icon="📊", layout="centered"
)

with st.sidebar:
    st.markdown("### 🎓 Project Details")
    st.markdown("**Project Title:** AI Resume Analyzer")
    st.markdown("**Semester:** 7th Semester B.Tech")
    st.write("---")

st.title("📊 AI Resume Analyzer & Parser")
st.markdown("### *7th Semester Engineering Project*")
st.write("---")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF format only)", type=["pdf"]
)
job_description = st.text_area(
    "Paste Job Description Here",
    height=150,
    placeholder="Looking for a Fresh Graduate / Junior Software Developer proficient in Python, SQL, Git, and OOP concepts...",
)

if st.button("🚀 Analyze and Match Resume"):
    if uploaded_file and job_description:
        with st.spinner("Analyzing text patterns..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            info = extract_resume_info(resume_text)
            match_score = calculate_match_score(resume_text, job_description)

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
                    st.success(", ".join(info["Skills"]))
                else:
                    st.warning("No beginner developer skills detected.")

            st.write("---")

            st.subheader("💡 Course Recommendations to Boost Match Score")
            missing_skills, rec_courses = get_course_recommendations(
                info["Skills"], job_description, match_score
            )

            if match_score < 50.0 or rec_courses:
                if missing_skills:
                    st.info(
                        f"Recommended skill areas to focus on: **{', '.join(missing_skills)}**"
                    )
                for i, course in enumerate(rec_courses, 1):
                    st.write(f"**{i}.** {course}")
            else:
                st.success(
                    "🎉 Great job! Your entry-level skill profile matches the core requirements of this job description."
                )
    else:
        st.error(
            "⚠️ Please provide both the resume PDF and the job description text."
        )
