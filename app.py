
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
model = joblib.load("student_model.pkl")

# Website heading
st.title("🎓 AI-Based Student Performance")
st.write(
    "Predict student final grade using Machine Learning."
)

st.divider()

st.subheader("📝 Enter Student Details")

# Input fields
g1 = st.number_input(
    "Previous Grade 1",
    min_value=0, max_value=20, value=10
)

g2 = st.number_input(
    "Previous Grade 2",
    min_value=0, max_value=20, value=12
)

studytime = st.selectbox(
    "Weekly Study Time",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours"
    }[x]
)

failures = st.number_input(
    "Past Class Failures",
    min_value=0, max_value=4, value=0
)

absences = st.number_input(
    "Number of Absences",
    min_value=0, max_value=93, value=2
)

# Prediction
if st.button("🔍 Predict Final Grade", type="primary"):

    student = pd.DataFrame(
        [[g1, g2, studytime, failures, absences]],
        columns=[
            "G1", "G2", "studytime",
            "failures", "absences"
        ]
    )

    prediction = float(model.predict(student)[0])
    prediction = max(0, min(20, prediction))

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric(
        label="Estimated Final Grade",
        value=f"{prediction:.1f} / 20"
    )
    
    # Create data for the chart
    chart_data = pd.DataFrame({
        "Grade Type": [
            "Previous Grade 1",
            "Previous Grade 2",
            "Predicted Final Grade"
        ],
        "Marks": [g1, g2, prediction]
    })

    # Display the chart
    st.subheader("📊 Student Performance Chart")

    st.bar_chart(
        chart_data.set_index("Grade Type")
    )


    # Performance category
    if prediction < 10:
        category = "Needs Improvement"
        st.warning(
            "Focus on daily revision and practise more questions."
        )
    elif prediction < 15:
        category = "Average Performance"
        st.info(
            "Continue regular study and revise difficult topics."
        )
    else:
        category = "Good Performance"
        st.success(
            "Great! Continue your study routine."
        )

    st.write("**Performance Category:**", category)

    # Grade chart
    chart_data = pd.DataFrame({
        "Grade Type": [
            "Previous Grade 1",
            "Previous Grade 2",
            "Predicted Final Grade"
        ],
        "Marks": [g1, g2, prediction]
    })

    st.subheader("📈 Student Grade Chart")
    st.bar_chart(
        chart_data.set_index("Grade Type")
    )

    # Attendance suggestion
    if absences > 10:
        st.write(
            "💡 Suggestion: Try to improve class attendance."
        )

    st.caption(
        "Note: This is an estimate, not a guaranteed exam result."
    )
