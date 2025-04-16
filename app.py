# app.py
'''
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load(open('Dragon.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            int(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            int(request.form['RAD']),
            int(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]

        prediction = model.predict([features])[0]
        prediction = round(prediction, 2)
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
'''
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load(open('Dragon.joblib', 'rb'))

# Static location-to-feature mapping
location_data = {
    'Allston':     {'CRIM': 0.06, 'ZN': 12.5, 'INDUS': 7.5, 'NOX': 0.45, 'DIS': 4.2, 'RAD': 5, 'TAX': 300, 'PTRATIO': 16.5, 'B': 390.0, 'LSTAT': 12.0},
    'Brookline':   {'CRIM': 0.02, 'ZN': 25.0, 'INDUS': 2.5, 'NOX': 0.38, 'DIS': 5.0, 'RAD': 3, 'TAX': 250, 'PTRATIO': 14.5, 'B': 395.0, 'LSTAT': 6.5},
    'Cambridge':   {'CRIM': 0.04, 'ZN': 22.0, 'INDUS': 3.0, 'NOX': 0.42, 'DIS': 4.5, 'RAD': 4, 'TAX': 280, 'PTRATIO': 15.2, 'B': 392.0, 'LSTAT': 9.0},
    'Dorchester':  {'CRIM': 0.09, 'ZN': 0.0,  'INDUS': 10.0,'NOX': 0.55, 'DIS': 3.0, 'RAD': 6, 'TAX': 330, 'PTRATIO': 18.0, 'B': 370.0, 'LSTAT': 14.5},
    'Fenway':      {'CRIM': 0.03, 'ZN': 15.0, 'INDUS': 5.0, 'NOX': 0.40, 'DIS': 5.2, 'RAD': 2, 'TAX': 270, 'PTRATIO': 13.8, 'B': 394.0, 'LSTAT': 7.5},
    'Jamaica Plain':{'CRIM': 0.07,'ZN': 10.0, 'INDUS': 6.5, 'NOX': 0.48, 'DIS': 3.8, 'RAD': 5, 'TAX': 310, 'PTRATIO': 15.8, 'B': 391.0, 'LSTAT': 10.0},
    'Roxbury':     {'CRIM': 0.1,  'ZN': 5.0,  'INDUS': 12.0,'NOX': 0.6,  'DIS': 2.5, 'RAD': 7, 'TAX': 340, 'PTRATIO': 18.5, 'B': 360.0, 'LSTAT': 15.0},
    'South Boston':{'CRIM': 0.05,'ZN': 18.0, 'INDUS': 4.0, 'NOX': 0.43, 'DIS': 4.7, 'RAD': 3, 'TAX': 290, 'PTRATIO': 14.9, 'B': 393.0, 'LSTAT': 8.2},
    'West Roxbury':{'CRIM': 0.03,'ZN': 20.0, 'INDUS': 3.5, 'NOX': 0.41, 'DIS': 4.9, 'RAD': 4, 'TAX': 275, 'PTRATIO': 14.2, 'B': 395.0, 'LSTAT': 6.0}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    price = None
    if request.method == 'POST':
        location = request.form['location']
        bedrooms = float(request.form['bedrooms'])
        built_year = int(request.form['built_year'])
        near_river = 1 if request.form['near_river'] == 'yes' else 0
        age = 2025 - built_year

        loc_feat = location_data[location]
        features = [
            loc_feat['CRIM'], loc_feat['ZN'], loc_feat['INDUS'], near_river,
            loc_feat['NOX'], bedrooms, age, loc_feat['DIS'],
            loc_feat['RAD'], loc_feat['TAX'], loc_feat['PTRATIO'],
            loc_feat['B'], loc_feat['LSTAT']
        ]

        prediction = model.predict([features])
        price = round(prediction[0], 2)

    return render_template('index.html', price=price)
if __name__ == '__main__':
    app.run(debug=True)
