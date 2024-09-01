from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

tsunami_dataset = pd.read_csv("tsunami_dataset.csv")
tsunami_dataset.dropna(subset=['EVENT_VALIDITY'], inplace=True)

X = tsunami_dataset[['YEAR', 'COUNTRY']]
y = tsunami_dataset['EVENT_VALIDITY']

numeric_features = ['YEAR']
categorical_features = ['COUNTRY']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

clf.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        country = request.form['country']
        year = int(request.form['year'])

        if (tsunami_dataset['COUNTRY'] == country).any() and (tsunami_dataset['YEAR'] == year).any():
            input_data = pd.DataFrame({'YEAR': [year], 'COUNTRY': [country]})
            predicted_event_validity = clf.predict(input_data)

            if year > 2023 and predicted_event_validity[0] == "Definite Tsunami":
                management_procedures = [
                    "Evacuate coastal areas immediately.",
                    "Alert emergency services and volunteers.",
                    "Prepare emergency shelters on higher ground.",
                    "Ensure access to emergency supplies such as food, water, and medical supplies.",
                    "Implement evacuation plans for coastal communities.",
                    "Monitor weather forecasts and warnings closely.",
                    "Coordinate with neighboring regions for assistance.",
                    "Establish communication channels for updates and alerts.",
                    "Arrange transportation for vulnerable populations.",
                    "Activate tsunami warning systems."
                ]
                return render_template('result.html', prediction=predicted_event_validity[0], management=management_procedures)
            else:
                return render_template('result.html', prediction=predicted_event_validity[0])
        else:
            return render_template('result.html', prediction="Very Doubtful Tsunami")

if __name__ == '__main__':
    app.run(debug=True, port=9500)
