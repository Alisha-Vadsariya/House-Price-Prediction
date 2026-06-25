# flask, scikit-learn, pandas, pickle-mixin
# import required libraries
import pickle
import pandas as pd
from flask import Flask, render_template, request

# create Flask application instance
app = Flask(__name__)

# read cleaned data
df_train = pd.read_csv('Cleaned_data_01.csv')

# load trained model from pickle file
pipe = pickle.load(open("RidgeModel_01.pkl", "rb"))


# define a route for the home page
@app.route('/')
def index():
    # get unique locations from the training data
    locations = sorted(df_train['location'].unique())
    # render the home page template with the unique locations as a dropdown
    return render_template('index.html', locations=locations)


# define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # get user inputs from the web form
    location = request.form.get('location')
    sqft = request.form.get('sqft')
    BHK = request.form.get('BHK')
    bath = request.form.get('bath')

    # validate user inputs and create input DataFrame
    if location and sqft and BHK and bath:
        sqft = float(sqft)
        BHK = float(BHK)
        bath = float(bath)
        input_df = pd.DataFrame({'total_sqft': [sqft], 'bath': [bath], 'BHK': [BHK], 'location': [location]})

        # make a prediction using the input DataFrame and the trained model
        PREDICTION = int(pipe.predict(input_df)[0] * 1e5)
        # return the prediction as a string
        return str(PREDICTION)
    else:
        # return an error message if the user inputs are invalid
        return "Please enter valid input values"


# start the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5001)
