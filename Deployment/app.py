from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import io
from functions import process_csv

app = Flask(__name__, template_folder='template')

rf_model = joblib.load('/media/prince/5A4E832F4E83034D/Fraud-Detector-ML/Models/v4/model_v4_rf.pkl')
xgb_model = joblib.load('/media/prince/5A4E832F4E83034D/Fraud-Detector-ML/Models/v4/model_v4_xg.pkl')
lgb_model = joblib.load('/media/prince/5A4E832F4E83034D/Fraud-Detector-ML/Models/v4/model_v4_lgb.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('fileToUpload')
    if not file or not file.filename.endswith('.csv'):
        return "Invalid CSV file", 400

    model_name = request.form.get('Model')
    options = request.form.getlist('options')

    if model_name == 'XG Boost':
        model = xgb_model
    elif model_name == 'RF_Clf':
        model = rf_model
    elif model_name == 'Light GB':
        model = lgb_model
    else:
        return "No valid model selected", 400

    try:
        df = process_csv(file)

        feature_cols = [c for c in df.columns if c != 'transaction_id']
        X = df[feature_cols]

        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)

        df['fraud_probability'] = (proba * 100)
        df['fraud_prediction'] = pred

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions.csv'
        )

    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)