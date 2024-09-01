import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv("pacific.csv")

# Preprocess latitude, longitude, and year columns
data['Latitude'] = data['Latitude'].str.rstrip('NS').astype(float)
data['Longitude'] = data['Longitude'].str.rstrip('EW').astype(float)
data['Year'] = pd.to_datetime(data['Date']).dt.year

# Feature selection
X = data[['Latitude', 'Longitude', 'Year']]
y = data['Status']  # Target variable

# Train the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X, y)

# Example usage: predicting status based on user input
user_latitude = float(input("Enter latitude (e.g., 20.6N): ").rstrip('NS'))
user_longitude = float(input("Enter longitude (e.g., 100.4W): ").rstrip('EW'))
user_year = int(input("Enter year (e.g., 2024): "))
user_input = [[user_latitude, user_longitude, user_year]]
predicted_status = classifier.predict(user_input)
print("Predicted status:", *predicted_status)
