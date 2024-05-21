import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data from CSV
data = pd.read_csv('falling_data.csv')

# Split data into features (X) and labels (y)
X = data[['Angle_1', 'Angle_2', 'Angle_3']]
y = data['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib

# Assuming model is your trained scikit-learn model
# Train your model (replace this with your actual training code)
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(rf_classifier, 'fall_detection_model.pkl')