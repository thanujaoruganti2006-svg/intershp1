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
    st.dataframe(classification_results.round(3), use_container_width=True)
    st.write("Confusion Matrix - Random Forest")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.caption("Pass is defined as a final score of 40 or above in this sample project.")

elif page == "Student Clustering":
    st.subheader("Student Performance Clustering")
    cluster_features = ["attendance", "study_hours", "final_score"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustered = df.copy()
    clustered["cluster"] = kmeans.fit_predict(scaled)
    st.write("Average values for each cluster")
    st.dataframe(clustered.groupby("cluster")[cluster_features].mean().round(2))
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        clustered["study_hours"], clustered["final_score"],
        c=clustered["cluster"], cmap="viridis", alpha=0.7
    )
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Score")
    ax.set_title("Student Performance Clusters")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    st.pyplot(fig)

elif page == "Course Recommendation":
    st.subheader("Course Recommendation System")
    courses = {
        "Python Programming": "python coding programming basics software development",
        "Machine Learning": "machine learning artificial intelligence prediction models",
        "Data Science": "data science statistics analytics visualization pandas",
        "Cybersecurity": "cybersecurity network security threat detection",
        "Cloud Computing": "cloud computing aws azure deployment storage",
        "Deep Learning": "deep learning neural networks tensorflow keras",
        "Natural Language Processing": "nlp text processing sentiment chatbot language",
        "Computer Vision": "computer vision image processing opencv images"
    }
    interest = st.text_area("Describe your interests", "I am interested in Python coding and artificial intelligence")
    if st.button("Recommend Courses", type="primary"):
        if interest.strip():
            names = list(courses.keys())
            descriptions = list(courses.values())
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(descriptions + [interest])
            scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
            top = scores.argsort()[::-1][:3]
            for idx in top:
                st.write(f"**{names[idx]}** — Similarity score: {scores[idx]:.2f}")
        else:
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
    st.subheader("Student Support Chatbot")
    def chatbot(question):
        question = question.lower()
        if "attendance" in question:
            return "Maintain good attendance and attend classes regularly."
        elif "marks" in question or "score" in question:
            return "Review your academic performance and focus on areas that need improvement."
        elif "python" in question:
            return "Start with variables, loops, functions, lists and dictionaries."
        elif "course" in question:
            return "You can explore Python, Data Science, Machine Learning and Cybersecurity."
        return "I can answer questions about attendance, marks, Python and learning courses."
    question = st.text_input("Ask a question")
    if st.button("Send"):
        if question.strip():
            st.info(chatbot(question))
        else:
            st.warning("Please type a question.")

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
