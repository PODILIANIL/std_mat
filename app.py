import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model, scaler, and label encoders
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load label encoders for categorical features
# You need to save your label_encoders dictionary to a file as well.
# For now, let's assume you have a way to re-create or load them.
# In a real deployment, you would save this dictionary as well.
# For demonstration, we'll create dummy encoders, but this needs to be accurate.

# NOTE: For deployment, ensure you save 'label_encoders' dictionary 
# from your notebook (e.g., using joblib.dump) and load it here.
# For example: label_encoders = joblib.load('label_encoders.pkl')

# This part needs to be accurately reflected from your notebook's label encoding step
# Define categorical columns based on your dataset
categorical_cols = [
    'school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
    'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
    'internet', 'romantic'
]

# Create dummy label encoders for the app. In a real scenario, load the trained ones.
# This part is highly dependent on your actual label_encoders from the notebook.
# If you saved label_encoders as a dictionary, load it like: 
# label_encoders = joblib.load('label_encoders.pkl')
# For this example, let's create a placeholder (you need to adapt this):
# A more robust way is to save the actual label_encoders dictionary from your training notebook.
# Example: joblib.dump(label_encoders, 'label_encoders.pkl')
# Then load here: label_encoders = joblib.load('label_encoders.pkl')

# Placeholder for label_encoders - YOU MUST REPLACE THIS WITH YOUR ACTUAL SAVED LABEL_ENCODERS
# Based on your notebook, all object columns were label encoded.
# You need to save this dictionary to a pickle file and load it here.
# For example, in your notebook after `label_encoders = {}` loop:
# joblib.dump(label_encoders, 'label_encoders.pkl')
# And here:
# label_encoders = joblib.load('label_encoders.pkl')

# If you do not save the label_encoders, you would need to create them
# from a sample of your training data here, which is error-prone.
# Let's assume you saved it as 'label_encoders.pkl'

# For the purpose of this example, we will assume you have a way to correctly
# initialize these or load them, as the exact mapping is critical.
# As a fallback, we'll create simple ones, but this needs to be fixed for real deployment.
class MockLabelEncoder:
    def fit_transform(self, X):
        unique_values = sorted(list(set(X)))
        self.mapping = {val: i for i, val in enumerate(unique_values)}
        return [self.mapping[x] for x in X]
    def transform(self, X):
        return [self.mapping.get(x, -1) for x in X] # -1 for unseen values
    def inverse_transform(self, X):
        inverse_mapping = {v: k for k, v in self.mapping.items()}
        return [inverse_mapping.get(x, 'Unknown') for x in X]

label_encoders = {
    col: MockLabelEncoder() for col in categorical_cols
}

# A more realistic approach would be to load the actual label_encoders
# Example of how you would load them if saved:
# try:
#     label_encoders = joblib.load('label_encoders.pkl')
#     print('Label encoders loaded successfully.')
# except FileNotFoundError:
#     print('label_encoders.pkl not found. Ensure you save it during training.')
#     # Handle the error or re-create them if possible

# Expected order of features based on your X.columns after dropping 'G3'
# You should verify this order from your training data (df.drop('G3', axis=1).columns)
feature_columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
                   'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
                   'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
                   'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                   'absences', 'G1', 'G2']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Convert data to appropriate types
    # Ensure all numerical features are converted to int/float
    # and categorical features are handled by label encoders
    processed_data = {}
    for col in feature_columns:
        value = data.get(col)
        if col in categorical_cols:
            # Apply label encoding
            # This assumes your MockLabelEncoder can handle transform for a single value
            # A robust way is to transform a list [value] and take the first element
            le = label_encoders[col]
            processed_data[col] = le.transform([value])[0]
        else:
            # Convert numerical features
            try:
                processed_data[col] = int(value) if value.isdigit() else float(value)
            except (ValueError, TypeError):
                processed_data[col] = 0 # Default or error handling

    # Create a DataFrame in the correct feature order
    input_df = pd.DataFrame([processed_data], columns=feature_columns)
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[:, 1][0] # Probability of 'Pass'

    result = 'Pass' if prediction == 1 else 'Fail'
    probability = f'{probability:.2f}'

    return render_template('result.html', prediction=result, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)