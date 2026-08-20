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
    """Safely extracts raw text preserving line breaks for contact metadata."""
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
    """Extracts candidate name by inspecting line structure."""
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

    # Normalize punctuation for clean skill detection
    normalized_for_skills = re.sub(r"[^\w\s+]", " ", clean_single_line.lower())
    extracted_skills = []

    for skill in SKILL_DATABASE.keys():
        pattern = (
            r"(?i)(?<![a-zA-Z0-9#+])" + re.escape(skill) + r"(?![a-zA-Z0-9#+])"
        )
        if re.search(pattern, normalized_for_skills):
            extracted_skills.append(skill.lower())

    return {
        "Name": name,
        "Email": email[0] if email else "Not Found",
        "Phone": phone,
        "Skills": list(set(extracted_skills)),
        "NormalizedText": clean_single_line,
    }


def calculate_match_score(resume_text, job_desc_text, resume_skills):
    job_desc_clean = re.sub(r"[^\w\s+]", " ", job_desc_text.lower())
    resume_text_clean = re.sub(r"[^\w\s+]", " ", resume_text.lower())

    # Build canonical skill set for candidate
    normalized_resume_skills = set()
    for s in resume_skills:
        s_lower = s.lower().strip()
        canonical_s = SYNONYMS.get(s_lower, s_lower)
        normalized_resume_skills.add(canonical_s)

    # Extract required skills from Job Description
    raw_required_skills = []
    for skill in SKILL_DATABASE.keys():
        pattern = (
            r"(?i)(?<![a-zA-Z0-9#+])" + re.escape(skill) + r"(?![a-zA-Z0-9#+])"
        )
        if re.search(pattern, job_desc_clean):
            raw_required_skills.append(skill)

    # Deduplicate canonical required skills
    canonical_required = set()
    for req in raw_required_skills:
        canonical = SYNONYMS.get(req, req)
        canonical_required.add(canonical)

    # ACCURATE PARTIAL SCORING CALCULATION
    if canonical_required:
        matched_count = sum(
            1 for req in canonical_required if req in normalized_resume_skills
        )
        skill_score = (matched_count / len(canonical_required)) * 100
    else:
        # If JD has no mapped skills from database, evaluate candidate skill presence in JD
        if normalized_resume_skills:
            found_count = sum(
                1 for s in normalized_resume_skills if s in job_desc_clean
            )
            skill_score = (found_count / len(normalized_resume_skills)) * 100
        else:
            skill_score = 0.0

    # TF-IDF Calculation
    try:
        documents = [resume_text_clean, job_desc_clean]
        vectorizer = TfidfVectorizer(
            stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        tfidf_matrix = vectorizer.fit_transform(documents)
        vector_sim = float(
            cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        )
        tfidf_score = vector_sim * 100
    except Exception:
        tfidf_score = 0.0

    # FINAL PROPORTIONAL SCORING LOGIC
    if canonical_required:
        if skill_score == 100.0:
            # Full Match: Guarantee high score 95-100%
            final_score = max(95.0, (100.0 * 0.85) + (tfidf_score * 0.15))
        else:
            # Partial Match: Scale strictly proportional to matched skill percentage
            final_score = (skill_score * 0.85) + (tfidf_score * 0.15)
    else:
        final_score = (skill_score * 0.50) + (tfidf_score * 0.50)

    return min(round(final_score, 2), 100.0)


def get_course_recommendations(resume_skills, job_desc_text):
    job_text_clean = re.sub(r"[^\w\s+]", " ", job_desc_text.lower())

    normalized_resume_skills = set()
    for s in resume_skills:
        s_lower = s.lower().strip()
        canonical_s = SYNONYMS.get(s_lower, s_lower)
        normalized_resume_skills.add(canonical_s)

    missing_skills = []
    recommendations = []

    for skill, course in SKILL_DATABASE.items():
        pattern = (
            r"(?i)(?<![a-zA-Z0-9#+])" + re.escape(skill) + r"(?![a-zA-Z0-9#+])"
        )
        if re.search(pattern, job_text_clean):
            canonical_skill = SYNONYMS.get(skill, skill)
            if canonical_skill not in normalized_resume_skills:
                if course not in recommendations:
                    missing_skills.append(skill.title())
                    recommendations.append(course)

    if not missing_skills and not recommendations:
        core_fallback_skills = [
            "python",
            "java",
            "sql",
            "git",
            "data structures",
            "rest api",
        ]
        for skill in core_fallback_skills:
            canonical_skill = SYNONYMS.get(skill, skill)
            if canonical_skill not in normalized_resume_skills:
                course = SKILL_DATABASE[skill]
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
    st.markdown("**Project Title:** AI Resume Analyzer & Parser ")
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
                            st.warning("No beginner developer skills detected.")

                    st.write("---")

                    st.subheader("💡 Skill Upgrade & Course Recommendations")
                    missing_skills, rec_courses = get_course_recommendations(
                        info["Skills"], job_description
                    )

                    if rec_courses:
                        st.info(
                            f"Missing Skill Areas to Focus On: **{', '.join(missing_skills)}**"
                        )
                        for i, course in enumerate(rec_courses, 1):
                            st.write(f"**{i}.** {course}")
                    elif match_score >= 85.0:
                        st.success(
                            "🎉 Exceptional match! Your skill profile covers core requirements detected in this job description."
                        )
                    else:
                        st.warning(
                            "⚠️ Low score match. Try adding more relevant technical skills, project details, and keywords from the job description to your resume."
                        )
            except Exception as e:
                st.error(f"⚠️ Error parsing PDF file: {str(e)}")
