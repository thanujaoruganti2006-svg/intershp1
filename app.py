
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Student Analytics", page_icon="🎓", layout="wide")

@st.cache_data
def create_dataset():
    np.random.seed(42)
    n = 600
    data = pd.DataFrame({
        "attendance": np.random.randint(40, 101, n),
        "study_hours": np.random.randint(1, 10, n),
        "assignment_score": np.random.randint(30, 101, n),
        "previous_score": np.random.randint(30, 101, n),
        "quiz_score": np.random.randint(30, 101, n)
    })
    noise = np.random.normal(0, 5, n)
    data["final_score"] = (
        0.25 * data["attendance"]
        + 2.0 * data["study_hours"]
        + 0.20 * data["assignment_score"]
        + 0.20 * data["previous_score"]
        + 0.15 * data["quiz_score"]
        + noise
    ).clip(0, 100)
    data["result"] = (data["final_score"] >= 40).astype(int)
    data = data.drop_duplicates()
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    return data

df = create_dataset()
features = ["attendance", "study_hours", "assignment_score", "previous_score", "quiz_score"]

@st.cache_resource
def train_regression(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["final_score"], test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R² Score": r2_score(y_test, pred)
    }
    return model, metrics

@st.cache_resource
def train_classifiers(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["result"], test_size=0.2,
        random_state=42, stratify=data["result"]
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    rows = []
    trained = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        trained[name] = clf
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0)
        })
    best = trained["Random Forest"]
    best_pred = best.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    return pd.DataFrame(rows).set_index("Model"), cm, best

regression_model, regression_metrics = train_regression(df)
classification_results, confusion_mat, best_classifier = train_classifiers(df)

st.title("🎓 AI-Based Student Analytics Dashboard")
st.write("Analyze student performance, predict scores, recommend courses, and explore AI-powered tools.")
st.caption("Note: This project currently uses a generated sample dataset of 600 students, not real student records.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a feature",
        [
            "Dashboard Overview",
            "Exploratory Data Analysis",
            "Predict Final Score",
            "Pass / Fail Classification",
            "Student Clustering",
            "Course Recommendation",
            "Feedback Sentiment",
            "Student Chatbot",
            "Suspicious Login Detection",
            "Image Processing"
        ]
    )
    st.divider()
    st.info("Select a feature above to use the application.")

if page == "Dashboard Overview":
    st.subheader("Project Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students", f"{len(df):,}")
    c2.metric("Average Final Score", f"{df.final_score.mean():.2f}")
    c3.metric("Pass Percentage", f"{100 * df.result.mean():.1f}%")
    st.subheader("Sample Student Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Dataset Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

elif page == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")
    st.write("Distribution of final scores")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["final_score"], bins=15, edgecolor="black")
    ax.set(title="Distribution of Final Scores", xlabel="Final Score", ylabel="Number of Students")
    st.pyplot(fig)
    st.write("Attendance compared with final score")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df["attendance"], df["final_score"], alpha=0.6)
    ax.set(title="Attendance vs Final Score", xlabel="Attendance (%)", ylabel="Final Score")
    st.pyplot(fig)
    st.write("Correlation heatmap")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif page == "Predict Final Score":
    st.subheader("Predict a Student's Final Score")
    st.write("Enter the student's details below.")
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.slider("Attendance (%)", 40, 100, 85)
        study_hours = st.slider("Study hours per day", 1, 9, 4)
        assignment_score = st.slider("Assignment score", 30, 100, 80)
    with c2:
        previous_score = st.slider("Previous score", 30, 100, 75)
        quiz_score = st.slider("Quiz score", 30, 100, 78)
    if st.button("Predict Final Score", type="primary"):
        new_student = pd.DataFrame([{
            "attendance": attendance, "study_hours": study_hours,
            "assignment_score": assignment_score,
            "previous_score": previous_score, "quiz_score": quiz_score
        }])
        predicted = float(regression_model.predict(new_student)[0])
        st.metric("Predicted Final Score", f"{predicted:.2f} / 100")
        st.write("Prediction is an estimate based on the generated training dataset.")

    st.subheader("Regression Model Evaluation")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{regression_metrics['MAE']:.2f}")
    m2.metric("RMSE", f"{regression_metrics['RMSE']:.2f}")
    m3.metric("R² Score", f"{regression_metrics['R² Score']:.3f}")

elif page == "Pass / Fail Classification":
    
    st.subheader("Pass / Fail Classification")

    st.write("Enter student academic details to predict the result.")

    c1, c2 = st.columns(2)

    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        study_hours = st.slider("Study Hours per Day", 0, 24, 5)
        assignment_score = st.slider("Assignment Score", 0, 100, 80)

    with c2:
        previous_score = st.slider("Previous Score", 0, 100, 75)
        quiz_score = st.slider("Quiz Score", 0, 100, 90)

    model_name = st.selectbox(
        "Choose Classification Model",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    if st.button("Predict Student Result", type="primary"):

        selected_model = {
            "Logistic Regression": None,
            "Decision Tree": None,
            "Random Forest": best_classifier
        }

        # Use the trained model selected by the user
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df["result"],
            test_size=0.2, random_state=42,
            stratify=df["result"]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        student = pd.DataFrame([{
            "attendance": attendance,
            "study_hours": study_hours,
            "assignment_score": assignment_score,
            "previous_score": previous_score,
            "quiz_score": quiz_score
        }])

        prediction = model.predict(student)[0]
        probability = model.predict_proba(student)[0][1] * 100

        if prediction == 1:
            st.success("Predicted Result: PASS")
        else:
            st.error("Predicted Result: FAIL")

        st.metric("Pass Probability", f"{probability:.2f}%")

    st.subheader("Classification Model Evaluation")
    st.dataframe(classification_results.round(3), use_container_width=True)

    st.write("Confusion Matrix - Random Forest")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


elif page == "Student Clustering":

    st.subheader("Interactive Student Performance Clustering")

    st.write("Change the number of clusters to explore different student groups.")

    cluster_features = ["attendance", "study_hours", "final_score"]

    # Select number of clusters
    n_clusters = st.slider(
        "Select Number of Student Groups",
        min_value=2,
        max_value=6,
        value=3
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    clustered = df.copy()
    clustered["cluster"] = kmeans.fit_predict(scaled)

    st.success(f"Students divided into {n_clusters} groups.")

    st.subheader("Cluster Summary")

    summary = clustered.groupby("cluster").agg(
        Students=("cluster", "count"),
        Average_Attendance=("attendance", "mean"),
        Average_Study_Hours=("study_hours", "mean"),
        Average_Final_Score=("final_score", "mean")
    ).round(2)

    st.dataframe(summary, use_container_width=True)

    st.subheader("Interactive Cluster Chart")

    x_axis = st.selectbox(
        "Choose X-axis",
        ["attendance", "study_hours", "final_score"]
    )

    y_axis = st.selectbox(
        "Choose Y-axis",
        ["final_score", "attendance", "study_hours"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    scatter = ax.scatter(
        clustered[x_axis],
        clustered[y_axis],
        c=clustered["cluster"],
        cmap="viridis",
        alpha=0.7
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("Student Performance Clusters")

    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

    st.subheader("View Students in a Group")

    selected_cluster = st.selectbox(
        "Select Student Group",
        sorted(clustered["cluster"].unique())
    )

    st.dataframe(
        clustered[clustered["cluster"] == selected_cluster],
        use_container_width=True
    )


elif page == "Course Recommendation":
  
    st.subheader("Personalized Course Recommendation System")

    st.write("Enter your interests and select your preferred learning area.")

    interest = st.text_area(
        "Describe your interests or skills",
        placeholder="Example: I know Python and I am interested in AI and Data Science"
    )

    category = st.selectbox(
        "Select your preferred category",
        ["All Categories", "Programming", "Artificial Intelligence",
         "Data Science", "Cybersecurity", "Cloud Computing"]
    )

    number = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=8,
        value=3
    )

    courses = {
        "Python Programming": ("Programming", "python coding programming basics software development"),
        "Machine Learning": ("Artificial Intelligence", "machine learning artificial intelligence prediction models"),
        "Data Science": ("Data Science", "data science statistics analytics visualization pandas"),
        "Cybersecurity": ("Cybersecurity", "cybersecurity network security threat detection"),
        "Cloud Computing": ("Cloud Computing", "cloud computing aws azure deployment storage"),
        "Deep Learning": ("Artificial Intelligence", "deep learning neural networks tensorflow keras"),
        "Natural Language Processing": ("Artificial Intelligence", "nlp text processing sentiment chatbot language"),
        "Computer Vision": ("Artificial Intelligence", "computer vision image processing opencv images")
    }

    if st.button("Recommend Courses", type="primary"):

        if interest.strip():

            filtered_courses = {
                name: details for name, details in courses.items()
                if category == "All Categories"
                or details[0] == category
            }

            names = list(filtered_courses.keys())
            descriptions = [details[1] for details in filtered_courses.values()]

            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(descriptions + [interest])

            scores = cosine_similarity(
                matrix[-1], matrix[:-1]
            ).flatten()

            top = scores.argsort()[::-1][:number]

            st.subheader("Your Recommended Courses")

            for idx in top:
                st.success(
                    f"{names[idx]} — Match Score: {scores[idx] * 100:.1f}%"
                )

        else:
            st.warning("Please enter your interests first.")

            st.warning("Please enter your interests.")

elif page == "Feedback Sentiment":
    st.subheader("Student Feedback Sentiment Analysis")
    feedback = st.text_area("Enter your feedback", "The course is very helpful and interesting")
    def simple_sentiment(text):
        positive_words = ["good", "great", "excellent", "helpful", "interesting", "easy", "useful", "clear"]
        negative_words = ["bad", "poor", "difficult", "hard", "boring", "confusing"]
        words = text.lower().split()
        positive_count = sum(word.strip(".,!?") in positive_words for word in words)
        negative_count = sum(word.strip(".,!?") in negative_words for word in words)
        if positive_count > negative_count:
            return "Positive"
        if negative_count > positive_count:
            return "Negative"
        return "Neutral"
    if st.button("Analyze Feedback", type="primary"):
        if feedback.strip():
            result = simple_sentiment(feedback)
            st.success(f"Sentiment: {result}")
        else:
            st.warning("Please enter feedback.")

elif page == "Student Chatbot":
    st.subheader("🎓 AI Student Support Chatbot")
    st.write("Ask questions about academics, attendance, marks, courses and study tips.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def chatbot(question):
        question = question.lower()

        if "attendance" in question:
            return "Maintain at least 75% attendance. Attend classes regularly and avoid unnecessary absences."

        elif "marks" in question or "score" in question:
            return "Review your previous marks, practise important questions and focus on subjects where you need improvement."

        elif "python" in question:
            return "Start learning Python with variables, data types, loops, functions, lists and dictionaries. Practise small programs daily."

        elif "course" in question:
            return "You can explore Python Programming, Data Science, Machine Learning, Cybersecurity and Cloud Computing."

        elif "study" in question or "exam" in question:
            return "Create a daily timetable, revise important topics, practise previous question papers and take short breaks."

        elif "project" in question:
            return "Choose a real-world problem, collect data, build your solution, test it and prepare a clear project presentation."

        elif "hello" in question or "hi" in question:
            return "Hello! Welcome to the Student Support Chatbot. How can I help you?"

        else:
            return "I can help with attendance, marks, Python, courses, exams and projects. Please ask me a question about these topics."

    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Type your question here...")

    if user_question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        answer = chatbot(user_question)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

elif page == "Suspicious Login Detection":
    st.subheader("Suspicious Login Detection")
    np.random.seed(7)
    login_data = pd.DataFrame({
        "login_attempts": np.random.randint(1, 6, 120),
        "login_hour": np.random.randint(6, 23, 120)
    })
    login_data.loc[:3, "login_attempts"] = [18, 25, 20, 30]
    login_data.loc[:3, "login_hour"] = [2, 3, 1, 4]
    anomaly_model = IsolationForest(contamination=0.05, random_state=42)
    login_data["prediction"] = anomaly_model.fit_predict(login_data)
    login_data["status"] = login_data["prediction"].map({1: "Normal", -1: "Suspicious"})
    c1, c2 = st.columns(2)
    c1.metric("Normal Logins", int((login_data.status == "Normal").sum()))
    c2.metric("Suspicious Logins", int((login_data.status == "Suspicious").sum()))
    st.dataframe(login_data.head(15), use_container_width=True)
    st.bar_chart(login_data["status"].value_counts())

elif page == "Image Processing":
    st.subheader("Image Upload and Grayscale Conversion")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        gray_image = image.convert("L")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Original Image")
            st.image(image, use_container_width=True)
        with c2:
            st.write("Grayscale Image")
            st.image(gray_image, use_container_width=True)
    else:
        st.info("Upload a PNG or JPG image to view its grayscale version.")

st.divider()
st.caption("AI Student Analytics Project | Streamlit application")
