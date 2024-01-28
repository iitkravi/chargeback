import pickle

from flask import Flask, render_template, request
from io import StringIO
import pandas as pd
import sklearn

app = Flask(__name__)

model= pickle.load(open('static/model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        csv_data = request.form.get('csv_data', '')
        if csv_data:
            csv_file = StringIO(csv_data)
            data = pd.read_csv("static/testdata.csv")  # Replace with your CSV file name
            data = data.fillna(0)  # Fill missing values with 0
            data = pd.get_dummies(data)
            x_data = data.drop("target", axis=1)
            print("Feature names during training:", model.feature_names_in_)
            df = pd.DataFrame(columns=[model.feature_names_in_])
            for col in df.columns:
                print(col)
            print("Feature names during prediction:", data.columns)
            x_data = pd.concat([df, x_data], ignore_index=True)
            result = model.predict(x_data)
            if result == 1:
                result = 'This is a chargeback transaction'
            else:
                result = 'This is a normal transaction'
    return render_template('index.html', data=result)


if __name__ == '__main__':
    app.run(debug=True)
