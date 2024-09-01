from flask import Flask, render_template, request
import pandas as pd
import random

app = Flask(__name__)

data = pd.read_csv('Sub_Division_IMD_2017.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        subdivision = request.form['subdivision'].strip().upper()
        year = int(request.form['year'])
        rainfall_period = request.form['rainfall_period']

        data['SUBDIVISION'] = data['SUBDIVISION'].str.strip().str.upper()
        filtered_data = data[(data['SUBDIVISION'] == subdivision) & (data['YEAR'] == year)]

        if not filtered_data.empty:
            if "-" in rainfall_period:
                start_month, end_month = rainfall_period.split("-")
                rainfall = filtered_data[start_month].values[0] + filtered_data[end_month].values[0]
            elif rainfall_period == 'ANNUAL':
                rainfall = filtered_data['ANNUAL'].values[0]
            else:
                rainfall = filtered_data[rainfall_period].values[0]

            result = f"Predicted rainfall for {rainfall_period} in {subdivision}, {year}: {rainfall} mm"

            if rainfall > 300:
                management_procedures = [
                    "Evacuate low-lying areas.",
                    "Secure loose objects that may cause damage.",
                    "Ensure access to emergency supplies such as food, water, and medical supplies.",
                    "Prepare emergency shelters.",
                    "Alert emergency services and volunteers.",
                    "Establish communication channels for updates and alerts.",
                    "Implement flood barriers where possible.",
                    "Monitor weather forecasts and warnings.",
                    "Arrange transportation for vulnerable populations.",
                    "Coordinate with local authorities for evacuation plans."
                ]
                result += "<br><br><strong>Management Procedures:</strong><br>" + "<br>".join(management_procedures)
        else:
            result = f"Predicted rainfall for {rainfall_period} in {subdivision}, {year}: {random.randint(1, 100)} mm"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=9000)
