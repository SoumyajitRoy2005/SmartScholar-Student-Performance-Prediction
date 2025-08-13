import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
import datetime
import sqlite3

# Page config
st.set_page_config(page_title="🎓 SmartScholar", layout="centered")

# Gradient Background CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #FF9A9E;
    background-image: linear-gradient(45deg, #FF9A9E 0%, #FECFEF 50%, #98C1FF 100%);
}
.navbar {
    background: linear-gradient(90deg, #FFB6C1, #87CEFA);
    padding: 14px;
    border-radius: 10px;
    display: flex;
    justify-content: space-evenly;
    align-items: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 999;
}
.navbar button {
    background: transparent;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
    cursor: pointer;
    padding: 8px 16px;
    border-radius: 8px;
    transition: background 0.3s;
}
.navbar button:hover {
    background: rgba(255, 255, 255, 0.2);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

def show_footer():
    st.markdown("""
        <hr style="margin-top: 50px;">
        <div style='text-align: center; color: black; font-size: 14px;'>
            &copy; 2025 Smart Scholar. All rights reserved.
        </div>
    """, unsafe_allow_html=True)

# --- Database Setup ---
conn = sqlite3.connect("reviews.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    message TEXT NOT NULL,
    stars INTEGER NOT NULL,
    time TEXT NOT NULL
)
""")
conn.commit()

def add_review(name, message, stars, time):
    c.execute("INSERT INTO reviews (name, message, stars, time) VALUES (?, ?, ?, ?)",
              (name, message, stars, time))
    conn.commit()

def get_reviews():
    c.execute("SELECT id, name, message, stars, time FROM reviews ORDER BY id DESC")
    return c.fetchall()

# --- Session State Initialization ---
if "step" not in st.session_state:
    st.session_state.step = 0  # Landing page
if "page" not in st.session_state:
    st.session_state.page = "Landing"
if "app_info" not in st.session_state:
    st.session_state.app_info = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "rating" not in st.session_state:
    st.session_state.rating = 0

# --- Page Map ---
page_map = {1: "Home 🏠", 2: "Prediction 📊", 3: "Study Planner", 4: "Feedback", 5: "Contact Us 📩"}

# --- Navigation Bar ---
def nav_bar():
    cols = st.columns(len(page_map))
    for i, (step, name) in enumerate(page_map.items()):
        if cols[i].button(name):
            st.session_state.step = step
            st.session_state.page = name
nav_bar()
st.write("")  # spacing after navbar

# --- Progress Bar Function ---
def progress_bar(current, total):
    percent = int((current / total) * 100)
    st.markdown(f"""
        <div style="background:#ddd;border-radius:20px;overflow:hidden;margin:10px 0;">
            <div style="width:{percent}%;
                        background:linear-gradient(90deg,#FFB6C1,#87CEFA);
                        padding:8px;
                        text-align:center;
                        color:black;
                        font-weight:bold;">
                Step {current}/{total}
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- Navigation Helpers ---
def next_page():
    if st.session_state.step < len(page_map):
        st.session_state.step += 1
        st.session_state.page = page_map[st.session_state.step]

def previous_page():
    if st.session_state.step > 0:
        st.session_state.step -= 1
        st.session_state.page = page_map.get(st.session_state.step, "Landing")

# --- Landing Page ---
if st.session_state.step == 0:
    st.markdown(
        "<h1 style='text-align: center; color: black;'>🎓 SMART SCHOLAR</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<h3 style='text-align:center;'>Welcome!</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Please select a page from the navigation bar above.</h3>", unsafe_allow_html=True)
    show_footer()

# --- Home Page ---
elif st.session_state.step == 1:
    progress_bar(1, 5)
    st.title("🏠 Home Page")

    intro_text = """
    Student performance prediction is an emerging field in educational data science that focuses on analyzing and forecasting a student’s academic success using machine learning and statistical techniques. By leveraging historical academic records, attendance, socio-economic factors, and behavioral patterns, predictive models can identify students at risk of underperforming and highlight factors contributing to their success. This approach helps educators, institutions, and policymakers make informed decisions, provide timely interventions, and design personalized learning plans to enhance overall academic outcomes. With advancements in artificial intelligence, student performance prediction is becoming an essential tool in building smart, data-driven educational systems.
    """
    st.markdown(f"<div style='background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; color: black; font-size: 18px; text-align: justify;'>{intro_text}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            previous_page()
    with col2:
        if st.button("➡ Next"):
            next_page()
    show_footer()

# --- Prediction Page ---
elif st.session_state.step == 2:
    progress_bar(2, 5)
    st.title("📊 Student Performance Predictor")

    # Load the saved model and encoders once
    try:
        with open("student_performance_model.pkl", "rb") as f:
            saved_data = pickle.load(f)
        model = saved_data['model']
        le_ext = saved_data['le_ext']
        le_par_ed = saved_data['le_par_ed']
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        le_ext = None
        le_par_ed = None

    def predict_performance(study_hours, attendance, previous_grade, extracurricular, parental_education):
        ext_val = 1 if extracurricular.lower() == 'yes' else 0
        if parental_education in le_par_ed.classes_:
            par_ed_val = le_par_ed.transform([parental_education])[0]
        else:
            par_ed_val = 0
        input_data = [[study_hours, attendance, previous_grade, ext_val, par_ed_val]]
        pred_score = model.predict(input_data)[0]
        pred_score = max(0, min(100, round(pred_score, 2)))
        result = 'Pass' if pred_score >= 40 else 'Fail'
        return pred_score, result


    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    name = st.text_input("Enter your name")
    if st.checkbox("Are you a student"):
        st.write("You are marked as a student")
    gender = st.selectbox("Gender", ["Select", "Male", "Female","Others"])
    study_hours = st.number_input("Study Hours per Day (hours spent studying at home)", min_value=0.0, max_value=24.0, value=0.00)
    attendance = st.number_input("Attendance (%) (attendance in school / university classes)", min_value=0, max_value=100, value=0)
    previous_grade = st.number_input("Previous Academic Grade (%) (percentage in last exam)", min_value=0, max_value=100, value=0)
    extracurricular = st.selectbox("Co-curricular Participation (school / university level education)", ["Select", "Yes", "No"])
    parental_education = st.selectbox("School / University Level Education", ["Select", "Primary School","Seconday School","High Seconday School","Bachelor's Degree", "Master’s Degree","Diploma", "Bachelor’s (Honours)","Phd"])

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded.")
        elif extracurricular == "Select" or parental_education == "Select":
            st.warning("Please select all dropdown values.")
        elif study_hours == 0 and attendance == 0 and previous_grade == 0:
            st.success("Predicted Final Score: 0")
            st.info("Result: Fail")
        else:
            try:
                predicted_score, pass_fail = predict_performance(study_hours, attendance, previous_grade, extracurricular, parental_education)
                st.success(f"Predicted Final Score: {predicted_score}")
                st.info(f"Result: {pass_fail}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")



    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            previous_page()
    with col2:
        if st.button("➡ Next"):
            next_page()

    show_footer()

# --- Study Planner Page ---
elif st.session_state.step == 3:
    progress_bar(3, 5)
    st.title("📚 Study Planning Suggestion")

    college_time = st.number_input("College Hours per Day", min_value=0.0, max_value=12.0, step=0.5)
    free_hours = st.number_input("Study Hours per Day (hours spent studying at home)", min_value=0.0, max_value=12.0, step=0.5)
    sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0, max_value=12.0, step=0.5)
    difficulty = st.selectbox("Difficulty Level of Subjects", ["Select", "Easy", "Moderate", "Hard"])

    valid_study = all([free_hours > 0, sleep_hours > 0, difficulty != "Select"])

    if st.button("Generate Study Plan"):
        if valid_study:
            plan = []
            current_time = 8  # Start at 8 AM

            # Sleep block
            sleep_start = (24 - sleep_hours + 8) % 24
            sleep_end = (sleep_start + sleep_hours) % 24

            # College block
            if college_time > 0:
                plan.append(f"{int(current_time):02d}:00 - {int((current_time + college_time) % 24):02d}:00 ➝ 🎓 College")
                current_time += college_time

            # Study sessions and breaks
            study_block = int(free_hours // 2)
            for i in range(study_block):
                start = int(current_time)
                end = (start + 1) % 24
                plan.append(f"{start:02d}:00 - {end:02d}:00 ➝ 📖 Study Session")
                current_time += 1

            if free_hours > 2:
                plan.append(f"{int(current_time):02d}:00 - {(int(current_time)+1)%24:02d}:00 ➝ ☕ Break/Relax")
                current_time += 1

            remaining_study = int(free_hours - study_block)
            for i in range(remaining_study):
                start = int(current_time)
                end = (start + 1) % 24
                plan.append(f"{start:02d}:00 - {end:02d}:00 ➝ 📝 Study Session")
                current_time += 1

            # Difficulty-based extra session
            if difficulty == "Easy":
                plan.append("Before bed ➝ 1 hour Revision.")
            elif difficulty == "Moderate":
                plan.append("Before bed ➝ 1.5 hours Problem Solving + Revision.")
            else:
                plan.append("Before bed ➝ 2 hours Deep Study + Mock Tests.")

            # Sleep block
            plan.append(f"{int(sleep_start):02d}:00 - {int(sleep_end):02d}:00 ➝ 🛌 Sleep")

            st.subheader("🕒 Your Optimized Daily Routine:")
            for item in plan:
                st.write(item)
        else:
            st.error("⚠ Fill all fields before generating the plan!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            previous_page()
    with col2:
        if st.button("➡ Next"):
            if valid_study:
                next_page()
            else:
                st.error("⚠ Fill all study fields before going next!")

    show_footer()

# --- Feedback / Review Page ---
if 'rating' not in st.session_state:
    st.session_state.rating = 0

# Step 4: Review form and display
if 'step' not in st.session_state:
    st.session_state.step = 4

if st.session_state.step == 4:
    st.title("💫 Review Us")

    st.write("Rating (Stars):")
    cols = st.columns(5)
    for i in range(5):
        if cols[i].button("⭐" if i < st.session_state.rating else "☆", key=f"star_{i}"):
            st.session_state.rating = i + 1
    st.write(f"Your rating: {st.session_state.rating} stars")

    with st.form("review_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        review_msg = st.text_area("Your Review")
        submit = st.form_submit_button("Submit Review")

        if submit:
            if not name.strip():
                st.warning("Please enter your name.")
            elif not review_msg.strip():
                st.warning("Please enter your review message.")
            elif st.session_state.rating == 0:
                st.warning("Please select a rating.")
            else:
                time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                add_review(name.strip(), review_msg.strip(), st.session_state.rating, time_now)
                st.success("Thank you for your review!")
                st.session_state.rating = 0  # reset stars

    reviews = get_reviews()

    for rev in reviews:
        rev_id, rev_name, rev_msg, rev_stars, rev_time = rev

        st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 15px; 
            margin-bottom: 10px; 
            background-color: #f9f9f9;
        ">
            <strong>{rev_name}</strong> <small style="color:gray;">({rev_time})</small><br>
            <p>Rating: {'⭐' * rev_stars}</p>
            <p>{rev_msg}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

        if st.button("Delete Review", key=f"del_{rev_id}"):
            c.execute("DELETE FROM reviews WHERE id = ?", (rev_id,))
            conn.commit()
            st.warning("Review deleted! Please refresh the page to see the changes.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            previous_page()
    with col2:
        if st.button("➡ Next"):
            next_page()
    show_footer()

# --- Contact Us Page ---
elif st.session_state.step == 5:
    progress_bar(5, 5)
    st.title("📩 Contact With Us")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(r"C:\Users\Rivan\Downloads\Student Performance Prediction Project\ARKA.jpg", caption="ARKA SADHUKHAN", width=150)
        if st.button("👤 View ARKA", key="arka_button"):
            st.markdown("""
                <div style="text-align: center; padding: 8px; background-color: #f0f8ff; border-radius: 10px;">
                    <strong>ARKA SADHUKHAN</strong><br>
                    UI/UX Designer<br>
                </div>
            """, unsafe_allow_html=True)
        if st.button("📩 Contact ARKA", key="contact_arka"):
            st.markdown("""
                <div style="text-align: center; padding: 8px; background-color: #f0f8ff; border-radius: 10px;">
                    <strong>CONTACT ME</strong><br>
                    <strong>aniarka7872@gmail.com</strong><br>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.image(r"C:\Users\Rivan\Downloads\Student Performance Prediction Project\MANAMI.jpg", caption="MANAMI MANNA", width=150)
        if st.button("👤 View Manami", key="manami_button"):
            st.markdown("""
                <div style="text-align: center; padding: 8px; background-color: #f0f8ff; border-radius: 10px;">
                    <strong>MANAMI MANNA</strong><br>
                    ML & Python Developer<br>
                </div>
            """, unsafe_allow_html=True)
        if st.button("📩 Contact MANAMI", key="contact_manami"):
            st.markdown("""
                <div style="text-align: center; padding: 8px; background-color: #f0f8ff; border-radius: 10px;">
                    <strong>CONTACT ME</strong><br>
                    <strong>manamimanna0@gmail.com</strong><br>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        st.image(r"C:\Users\Rivan\Downloads\Student Performance Prediction Project\SOUMYAJIT.jpg", caption="SOUMYAJIT ROY", width=150)
        if st.button("👤 View Soumyajit", key="soumyajit_button"):
            st.markdown("""
                <div style="text-align: center; padding: 8px; background-color: #f0f8ff; border-radius: 10px;">
                    <strong>SOUMYAJIT ROY</strong><br>
                    Frontend & Backend Developer
                </div>
            """, unsafe_allow_html=True)
        if st.button("📩 Contact SOUMYAJIT", key="contact_soumyajit"):
            st.markdown("""
                <div style="text-align: center; padding: 8px; background-color: #f0f8ff; border-radius: 10px;">
                    <strong>CONTACT ME</strong><br>
                    <strong>soumyajitroy0303@gmail.com</strong><br>
                </div>
            """, unsafe_allow_html=True)

    if st.button("⬅ Back"):
        previous_page()

    show_footer()

