from flask import Flask, request
import pickle
import os
import pandas as pd


with open('lr_clf.pickle', 'rb') as pfile:
    clf_loaded = pickle.load(pfile)

COLUMNS = ['State_FIPS', '116th_Congressional_District', '2017_NAICS_Code',
           'Number_of_Establishments', 'Employment', 'Employment_Noise_Flag',
           '1st_Quarter_Payroll_Noise_Flag', 'Annual_Payroll_Noise_Flag']


app = Flask(__name__)


# client
@app.route('/predict_single', methods=['GET'])
def predict_single():
    df = pd.DataFrame(columns=COLUMNS)
    df.loc[0, :] = [float(request.args[col]) for col in COLUMNS]
    return f'{clf_loaded.predict(df)[0].round():,}'


# client
@app.route('/predict', methods=['POST'])
def predict():
    """receives a json file from client,
     modified from a dataframe of X_text -->
     X_test.to_json(orient='Table')"""

    if request.is_json:
        req = request.get_json(force=True)
        df = pd.read_json(req, orient='records')
        return pd.DataFrame(clf_loaded.predict(df).round()).to_json(orient='records')


if __name__ == '__main__':
    # Heroku provides environment variable 'PORT' that should be listened on by Flask
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run(port=5000)
