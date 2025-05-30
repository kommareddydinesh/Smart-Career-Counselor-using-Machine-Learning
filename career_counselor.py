!pip install pandas numpy scikit-learn streamlit

!pip install streamlit
!pip install localtunnel
!streamlit run career_counselor.py &
# Tunnel to make it public
!npx localtunnel --port 8501


# Smart Career Counselor using AI & ML

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Sample dataset (you can replace this with real-world data)
data = {
    'Maths_Score': [85, 90, 70, 60, 50, 40, 30, 95, 88, 45],
    'Biology_Score': [30, 35, 80, 90, 95, 85, 88, 25, 33, 70],
    'Computer_Score': [95, 98, 60, 55, 45, 35, 25, 99, 96, 40],
    'Interest': ['Programming', 'Programming', 'Medicine', 'Medicine', 'Medicine', 'Biology', 'Biology', 'Programming', 'Programming', 'Biology'],
    'Career': ['Software Engineer', 'Software Engineer', 'Doctor', 'Doctor', 'Doctor', 'Biologist', 'Biologist', 'Software Engineer', 'Software Engineer', 'Biologist']
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical features
le_interest = LabelEncoder()
df['Interest'] = le_interest.fit_transform(df['Interest'])

le_career = LabelEncoder()
df['Career'] = le_career.fit_transform(df['Career'])

# Step 3: Define features and target
X = df[['Maths_Score', 'Biology_Score', 'Computer_Score', 'Interest']]
y = df['Career']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Get unique class labels from test and predictions
unique_labels = np.unique(np.concatenate((y_test, predictions)))
report = classification_report(y_test, predictions, labels=unique_labels, target_names=le_career.inverse_transform(unique_labels))
conf_matrix = confusion_matrix(y_test, predictions, labels=unique_labels)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# Step 7: Save model and encoders
pickle.dump(model, open('career_model.pkl', 'wb'))
pickle.dump(le_interest, open('le_interest.pkl', 'wb'))
pickle.dump(le_career, open('le_career.pkl', 'wb'))

# ---------------------------
# Streamlit Web Application
# ---------------------------

# Load model and encoders
model = pickle.load(open('career_model.pkl', 'rb'))
le_interest = pickle.load(open('le_interest.pkl', 'rb'))
le_career = pickle.load(open('le_career.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Smart Career Counselor", layout="centered")
st.title("\U0001F4DA Smart Career Counselor")
st.write("Enter your details below to get a personalized career recommendation:")

maths = st.slider("Maths Score", 0, 100, 50)
biology = st.slider("Biology Score", 0, 100, 50)
computers = st.slider("Computer Score", 0, 100, 50)
interest_input = st.selectbox("Interest Area", le_interest.classes_)

# Encode user input
interest_encoded = le_interest.transform([interest_input])[0]

# Predict button
if st.button("Predict Career"):
    user_data = np.array([[maths, biology, computers, interest_encoded]])
    prediction = model.predict(user_data)[0]
    career_output = le_career.inverse_transform([prediction])[0]
    st.success(f"\U0001F393 Recommended Career Path: **{career_output}**")

    # Show evaluation metrics
    st.subheader("Model Performance")
    st.text(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.code(report)

    # Confusion matrix heatmap
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le_career.inverse_transform(unique_labels), yticklabels=le_career.inverse_transform(unique_labels), cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Show user input data
    st.subheader("Your Input")
    st.json({
        "Maths Score": maths,
        "Biology Score": biology,
        "Computer Score": computers,
        "Interest Area": interest_input
    })
