import pickle

import numpy
from flask import Flask, render_template, request
from io import StringIO
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

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
            # data = pd.get_dummies(data)
            x_data = data.drop("target", axis=1)
            print("Feature names during training:", model.feature_names_in_)
            #df = pd.DataFrame(columns=[model.feature_names_in_])
            # for col in df.columns:
            #     print(col)
            # print("Feature names during prediction:", data.columns)
            encoder = LabelEncoder()
            encoder.classes_ = numpy.load('static/classes.npy', allow_pickle=True)
            x_data = x_data[['merchant_name', 'merchant_id', 'merchant_email', 'merchant_phone', 'customer_name',
                   'customer_card_number',
                   'customer_billing_address', 'city', 'zip_code', 'order_date', 'order_amount', 'order_currency',
                   'order_description', 'payment_authorization_code', 'payment_settlement_date', 'latitude',
                   'longitude',
                   'ip_address']].apply(encoder.fit_transform)
            #x_data = pd.concat([df, x_data], ignore_index=True)
            print(x_data)
            result = model.predict(x_data)
            print(result)
            if result == 1:
                result = 'This is a chargeback transaction'
            else:
                result = 'This is a normal transaction'
    return render_template('index.html', data=result)


if __name__ == '__main__':
    app.run(debug=True)
