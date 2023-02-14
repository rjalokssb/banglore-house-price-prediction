from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('Ridgemodel.pkl','rb'))
data = pd.read_csv('Cleaned_data.csv')
@app.route('/')
def home():
    location = sorted(data['location'].unique())
    return render_template('index.html', **locals())

@app.route('/predict', method=['POST', 'GET'])
def predict():
    location = float(request.form['location'])
    total_sqft = float(request.form['total_sqft'])
    bath = float(request.form['bath'])
    bhk = float(request.form['bhk'])
    input = pd.DataFrame([[location,total_sqft, bath, bhk]], columns =['location', 'total_sqft', 'bath', 'bhk'])
    result = model.predict(input)
    return render_template('index.html', **locals())



if __name__ == '__main__':
    app.run(debug=True)