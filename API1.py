import pandas as pd
from flask import Flask, request
import pickle
import os

with open('lr_clf.pickle', 'rb') as pfile:
    clf_loaded = pickle.load(pfile)

def url_to_x_test(params):
    split_params = params.split('&')
    df = pd.DataFrame({value.split('=')[0]: [value.split('=')[1]] for value in split_params})
    return df


app = Flask(__name__)

# client
@app.route('/predict_single/<params>', methods=['GET'])
def predict_single(params):
    """receives a url filled with all parameters,
    decoded for proper linear regression object expectations,
    sent to predict, one line at a time"""

    '''
    try the following:
    http://127.0.0.1:5000/predict_single/State_FIPS=02&116th_Congressional_District=5.0&2017_NAICS_Code=41&Number_of_Establishments=800&Employment=6128 
    &Employment_Noise_Flag=1&1st_Quarter_Payroll_Noise_Flag=1&Annual_Payroll_Noise_Flag=1
    '''

    X = url_to_x_test(params)
    return f'{clf_loaded.predict(X)[0].round():,}'


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
        app.run()