from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model',methods=['POST'])
def model():
    model = pickle.load(open('modelv1.pkl', 'rb'))
    
    features = [np.array([x for x in request.form.values()])]

    col_names = ['state', 'account_length', 'area_code', 'international_plan',
       'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
       'total_day_calls', 'total_day_charge', 'total_eve_minutes',
       'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
       'total_night_calls', 'total_night_charge', 'total_intl_minutes',
       'total_intl_calls', 'total_intl_charge',
       'number_customer_service_calls']
    
    user_df_input = pd.DataFrame(data=features,columns=col_names)

    for col in user_df_input.columns[user_df_input.dtypes == 'object']:
        user_df_input[col]=user_df_input[col].astype('category').cat.codes
        
    prediction = model.predict(user_df_input)

    try:
        if (prediction == 1):
            output = "The Customer will Churn"
        if (prediction == 0):
            output = "The Customer will not Churn"
    except:
        output = "Error"

    return render_template('index.html', result=output)

if __name__ == "__main__":
    app.run(debug=True)
