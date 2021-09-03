from flask import Flask, render_template, request,jsonify
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
    print(features)

    col_names = ['state', 'account_length', 'area_code', 'international_plan',
       'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
       'total_day_calls', 'total_day_charge', 'total_eve_minutes',
       'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
       'total_night_calls', 'total_night_charge', 'total_intl_minutes',
       'total_intl_calls', 'total_intl_charge',
       'number_customer_service_calls']
    
    features_df = pd.DataFrame(data=features,columns=col_names)

    for col in features_df.columns[features_df.dtypes == 'object']:
        features_df[col]=features_df[col].astype('category').cat.codes
        
    prediction = model.predict_proba(features_df)
    predict = model.predict(features_df)
    predict_prob = round(prediction[0][0],2)
    
    result={
         "predict_prob":predict_prob,
          }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
