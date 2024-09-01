from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)

data = pd.read_csv("eruptions.csv")
imputer = SimpleImputer(strategy='mean')
data[['latitude', 'longitude']] = imputer.fit_transform(data[['latitude', 'longitude']])
X = data[['latitude', 'longitude']]
y = data['eruption_category']
model = RandomForestClassifier()
model.fit(X, y)

def get_precaution_measures():
    # Define precaution measures based on the predicted eruption category
    precaution_measures = {
        "Confirmed Eruption": [
            "Evacuate immediately to a safe location.",
            "Follow evacuation routes provided by local authorities.",
            "Stay away from the affected area until it is declared safe."
            "Prepare an emergency kit with essential supplies."
            "Keep important documents and contact information handy."
            "Stay informed through local news and authorities' instructions."
            "Seek shelter in a sturdy building if evacuation is not possible."
            "Stay away from rivers and low-lying areas to avoid flooding."
            "Avoid using elevators during evacuation; use stairs instead."
        ]
    }
    
    # Default precaution measures for cases where the eruption category is not found
    default_precaution_measures = [
        "Follow safety protocols recommended by local authorities.",
        "Stay informed about the situation through reliable sources.",
        "Prepare an emergency kit with essential supplies.",
        "Stay away from the affected area until it is declared safe.",
        "Seek shelter in a sturdy building if evacuation is not possible."
    ]
    
    return precaution_measures, default_precaution_measures

def is_in_india(latitude, longitude):
    if latitude >= 8.4 and latitude <= 37.6 and longitude >= 68.1 and longitude <= 97.4:
        return True
    else:
        return False

def predict_eruption_category(latitude, longitude, start_year):
    if is_in_india(latitude, longitude):
        return "Uncertain Eruption", get_precaution_measures()[1]
    else:
        input_data = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
        prediction = model.predict(input_data)
        precaution_measures, default_precaution_measures = get_precaution_measures()
        return prediction[0], precaution_measures.get(prediction[0], default_precaution_measures)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_latitude = float(request.form['latitude'])
        user_longitude = float(request.form['longitude'])
        start_year = int(request.form['start_year'])

        predicted_category, precaution_measures = predict_eruption_category(user_latitude, user_longitude, start_year)
        return render_template('result.html', prediction=predicted_category, precautions=precaution_measures)

if __name__ == '__main__':
    app.run(debug=True, port=9700)
