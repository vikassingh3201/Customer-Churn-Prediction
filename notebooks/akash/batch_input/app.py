from flask import Flask, render_template, request,jsonify
import pickle
import ast
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model',methods=['POST'])
def model():
    model = pickle.load(open('artifacts/modelv1.pkl', 'rb'))

    if request.method == 'POST':
        batch_input = request.form['data']
        final = pd.DataFrame.from_dict(ast.literal_eval(batch_input))

        for col in final.columns[final.dtypes == 'object']:
            final[col]=final[col].astype('category').cat.codes
        
        predict_prob = model.predict_proba(final).tolist()
        prediction = model.predict(final).tolist()
        threshold_val=0.5

        arr=[]
        for i in range(0,10):
            temp={
            "predict":prediction[i],
            "predict_prob":round(predict_prob[i][1],2),
            "threshold":threshold_val
            }
            arr.append(temp)

        result = dict(enumerate(np.array(arr), 0))
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)