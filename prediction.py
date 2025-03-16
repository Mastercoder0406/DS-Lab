import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the trained model
model = load_model("mental_health_model.keras")

# Load the saved LabelEncoders
encoders = joblib.load("label_encoders.pkl")

# Features used in the model
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 
                'anonymity', 'leave', 'work_interfere']

# Standardizing the input format
def format_input(value, encoder, feature_name):
    """
    Ensures user input matches expected format by checking encoder classes.
    """
    value = value.strip().capitalize()  # Convert 'yes' -> 'Yes', 'no' -> 'No'
    print(f"🟢 Debug: {feature_name} accepted values: {encoder.classes_}")  # Show expected labels
    if value not in encoder.classes_:
        print(f"⚠️ Error: The input '{value}' is not recognized. Please enter a valid value from {list(encoder.classes_)}.")
        return None
    return value

# Function to take user input and make predictions
def predict_mental_health():
    print("\n🔹 Enter the following details for prediction:")
    
    # Take user input
    age = float(input("Enter your Age: "))
    
    # Ensure correct string input for categorical features
    gender = input("Enter your Gender (male/female/trans): ").strip().capitalize()
    family_history = format_input(input("Do you have a family history of mental illness? (yes/no): "), encoders['family_history'], 'family_history')
    benefits = format_input(input("Does your employer provide mental health benefits? (yes/no): "), encoders['benefits'], 'benefits')
    care_options = format_input(input("Do you have access to mental health care options? (yes/no): "), encoders['care_options'], 'care_options')
    anonymity = format_input(input("Can you discuss mental health anonymously at work? (yes/no): "), encoders['anonymity'], 'anonymity')
    leave = format_input(input("Are you comfortable taking leave for mental health? (yes/no): "), encoders['leave'], 'leave')
    work_interfere = format_input(input("Does work interfere with your mental health? (never/rarely/sometimes/often): "), encoders['work_interfere'], 'work_interfere')

    # Stop execution if any input was incorrect
    if None in [family_history, benefits, care_options, anonymity, leave, work_interfere]:
        print("⚠️ Please restart and enter valid inputs.")
        return

    # Convert categorical inputs to numerical using LabelEncoders
    input_data = [age]
    categorical_features = [gender, family_history, benefits, care_options, anonymity, leave, work_interfere]

    for i, feature in enumerate(categorical_features):
        encoder = encoders[feature_cols[i+1]]  # Retrieve encoder for the feature
        input_data.append(encoder.transform([feature])[0])  # Convert to numerical

    # Convert input data to NumPy array and reshape for model
    final_input = np.array([input_data]).reshape(1, -1)

    # Make prediction
    prediction_prob = model.predict(final_input)[0][0]
    output = round(prediction_prob * 100, 2)
    print(output)

    # Display the result
    if prediction_prob > 0.5:
        print(f"\n🔴 You might need mental health treatment. Probability: {output}%")
    else:
        print(f"\n🟢 You may not need mental health treatment. Probability: {output}%")

# Run the prediction function
if __name__ == "__main__":
    predict_mental_health()
