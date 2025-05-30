# Smart-Career-Counselor-using-Machine-Learning
A great name for your project could be:  "AI-Powered Smart Career Counselor"

# Overall Purpose:
This project is an AI-powered career counselor that predicts a suitable career path based on scores in Maths, Biology, Computer Science, and the user’s area of interest. It uses a machine learning model (Random Forest classifier) trained on sample data and a simple web app interface using Streamlit to interact with users.

# Step-by-step explanation:

**Step 1: Import necessary libraries**
pandas, numpy: For data manipulation and numerical operations.
sklearn: To create and evaluate the machine learning model.
pickle: To save and load the trained model and encoders.
streamlit: To build a simple interactive web app.
matplotlib, seaborn: For plotting the confusion matrix heatmap.

**Step 2: Prepare dataset**
You create a small dataset with:
Scores in Maths, Biology, and Computer Science.
User's interest area (Programming, Medicine, Biology).
Corresponding career paths (Software Engineer, Doctor, Biologist).

data = {
    'Maths_Score': [...],
    'Biology_Score': [...],
    'Computer_Score': [...],
    'Interest': [...],
    'Career': [...]
}
df = pd.DataFrame(data)
This dataset simulates training data for the model.

**Step 3: Encode categorical variables**
Machine learning models require numeric inputs, so you convert categorical columns into numbers:
Interest and Career columns are label encoded using LabelEncoder.

**Step 4: Define features and target**
X (features) = Scores + Interest (encoded)
y (target) = Career (encoded)

**Step 5: Split data**
Data split into training (80%) and testing (20%) to evaluate the model’s performance on unseen data.

**Step 6: Train the model**
A Random Forest Classifier is trained on the training data (X_train, y_train).
The model learns to predict career paths based on scores and interests.

**Step 7: Evaluate the model**
Make predictions on test data (X_test).
Calculate accuracy, classification report, and confusion matrix.
Fix applied: To handle cases where test data may not contain all classes, you dynamically extract unique classes from true and predicted labels for evaluation.

**Step 8: Save model and encoders**
Use pickle to save the trained model and label encoders so you can reuse them later without retraining.

**Step 9: Streamlit Web Application (Frontend)**
Load the saved model and encoders.
Create a user-friendly UI with:
1.Sliders to input scores (Maths, Biology, Computer Science).
2.Dropdown to select interest area.
Encode the user input (interest) the same way as the training data.
On pressing "Predict Career":
1.Run the ML model to predict career.
2.Display the predicted career.
3.Show model performance metrics (accuracy, classification report).
4.Show a confusion matrix heatmap using seaborn.
5.Display the user’s inputs for clarity.


# User flow on the app:
**1.User adjusts sliders for scores.**
**2.User picks their interest.**
**3.User clicks "Predict Career".**
**4.The app predicts and shows the recommended career and model stats visually.**


# Why this is useful:
**a.Demonstrates an end-to-end ML pipeline: data processing → model training → evaluation → deployment.**
**b.Uses simple data but scalable to bigger real datasets.**
**c.Streamlit interface allows anyone to get career suggestions interactively.**
