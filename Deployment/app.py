from flask import Flask, render_template, request, send_file, jsonify, session, redirect
import pandas as pd
import numpy as np
import joblib
import os
import uuid
import time
from functions import process_csv
from huggingface_hub import hf_hub_download

app = Flask(__name__, template_folder='template')

# FIX: Use a static secret key so sessions persist across multiple workers and server reloads!
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'fraud-sense-production-key-12345')

REPO_ID = 'mlwithprince/Fruad-Deteaction_models'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp_predictions')
os.makedirs(TEMP_DIR, exist_ok=True)

# Load Models
rf_model  = joblib.load(hf_hub_download(repo_id=REPO_ID, filename='mode_rf.pkl'))
xgb_model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename='model_xg.pkl'))
lgb_model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename='model_lgb.pkl'))

pr_curves = {
    'rf'  : joblib.load(hf_hub_download(repo_id=REPO_ID, filename='pr_curve_rf.pkl')),
    'xgb' : joblib.load(hf_hub_download(repo_id=REPO_ID, filename='pr_curve_xgb.pkl')),
    'lgb' : joblib.load(hf_hub_download(repo_id=REPO_ID, filename='pr_curve_lgb.pkl')),
}


# --- GARBAGE COLLECTION ---
def cleanup_old_files():
    now = time.time()
    for filename in os.listdir(TEMP_DIR):
        filepath = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(filepath) and os.stat(filepath).st_mtime < now - 43200:
            try:
                os.remove(filepath)
            except Exception:
                pass


# --- ML HELPERS ---
def get_threshold_from_pr(model_key, mode, value):
    curve     = pr_curves[model_key]
    precision = np.array(curve['precision'])
    recall    = np.array(curve['recall'])
    threshold = np.array(curve['thresholds'])
    if mode == 'precision':
        mask = precision >= value
        return float(threshold[-1]) if not mask.any() else float(threshold[mask][0])
    else:
        mask = recall >= value
        return float(threshold[0]) if not mask.any() else float(threshold[mask][-1])


def get_pr_at_threshold(model_key, t):
    curve          = pr_curves[model_key]
    precision      = np.array(curve['precision'])
    recall         = np.array(curve['recall'])
    thresholds_arr = np.array(curve['thresholds'])
    idx = int(np.clip(np.searchsorted(thresholds_arr, t, side='right') - 1, 0, len(precision) - 1))
    return round(float(precision[idx]), 4), round(float(recall[idx]), 4)


def run_core_model(model, X, threshold):
    proba = model.predict_proba(X)[:, 1]
    return proba, (proba >= threshold).astype(int)


def run_apex(X, threshold):
    rf_p  = rf_model.predict_proba(X)[:, 1]
    xgb_p = xgb_model.predict_proba(X)[:, 1]
    lgb_p = lgb_model.predict_proba(X)[:, 1]
    avg   = np.mean([rf_p, xgb_p, lgb_p], axis=0)
    std   = np.std([rf_p, xgb_p, lgb_p], axis=0)
    conf  = avg * (1 - std)
    return conf, (conf >= threshold).astype(int)


# --- ROUTES ---

@app.route('/')
def home():
    active_file_id = session.get('file_id')
    if active_file_id:
        if not os.path.exists(os.path.join(TEMP_DIR, f"{active_file_id}.csv")):
            session.pop('file_id', None)
            active_file_id = None
    return render_template('home.html', active_file_id=active_file_id)


@app.route('/pr_info', methods=['GET'])
def pr_info():
    model_key = request.args.get('model', 'xgb').lower()
    mode      = request.args.get('mode', 'threshold')
    value     = float(request.args.get('value', 50)) / 100.0
    if model_key not in pr_curves:
        return jsonify({'error': 'Invalid model'}), 400
    t = value if mode == 'threshold' else get_threshold_from_pr(model_key, mode, value)
    p, r = get_pr_at_threshold(model_key, t)
    return jsonify({'threshold': round(t, 4), 'precision': round(p * 100, 2), 'recall': round(r * 100, 2)})


@app.route('/predict', methods=['POST'])
def predict():
    cleanup_old_files()
    file = request.files.get('fileToUpload')
    if not file or not file.filename.endswith('.csv'):
        return jsonify({"error": "Please upload a valid .csv file"}), 400
    model_name = request.form.get('Model')
    if not model_name:
        return jsonify({"error": "No model selected"}), 400
    try:
        df           = process_csv(file)
        feature_cols = [c for c in df.columns if c != 'transaction_id']
        X            = df[feature_cols]

        if model_name in ('Core XGB', 'Core LGB', 'Core RF'):
            model_key_map = {'Core XGB': 'xgb', 'Core LGB': 'lgb', 'Core RF': 'rf'}
            model_map     = {'Core XGB': xgb_model, 'Core LGB': lgb_model, 'Core RF': rf_model}
            model_key     = model_key_map[model_name]
            model         = model_map[model_name]
            mode          = request.form.get('pr_mode', 'threshold')
            value         = float(request.form.get('pr_value', 50))
            t             = value / 100.0 if mode == 'threshold' else get_threshold_from_pr(model_key, mode, value / 100.0)
            proba, pred   = run_core_model(model, X, t)
            p, r          = get_pr_at_threshold(model_key, t)
            df.insert(0, 'fraud_prediction', pred)
            df.insert(0, 'fraud_probability', (proba * 100).round(2))
            df['threshold_used'] = round(t, 4)
            df['precision_at_t'] = round(p * 100, 2)
            df['recall_at_t']    = round(r * 100, 2)

        elif model_name == 'Apex 1.0':
            t              = float(request.form.get('threshold', '50')) / 100.0
            conf, pred     = run_apex(X, t)
            df.insert(0, 'fraud_prediction', pred)
            df.insert(0, 'fraud_probability', (conf * 100).round(2))
        else:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        file_id            = str(uuid.uuid4())
        filepath           = os.path.join(TEMP_DIR, f"{file_id}.csv")
        df.to_csv(filepath, index=False)
        session['file_id'] = file_id
        return jsonify({"status": "success", "file_id": file_id, "columns": df.columns.tolist()})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route('/cancel_session', methods=['POST'])
def cancel_session():
    file_id = session.get('file_id')
    if file_id:
        filepath = os.path.join(TEMP_DIR, f"{file_id}.csv")
        if os.path.exists(filepath):
            try: 
                os.remove(filepath)
                print(f"Successfully deleted {filepath}")
            except Exception as e: 
                print(f"Error deleting file: {e}")
        session.pop('file_id', None)
    else:
        print("No active file_id found in session to delete.")
    return jsonify({"status": "cleared"})


@app.route('/results', methods=['GET'])
def results_page():
    file_id = session.get('file_id')
    if not file_id: return redirect('/')
    filepath = os.path.join(TEMP_DIR, f"{file_id}.csv")
    if not os.path.exists(filepath): return redirect('/')
    columns = pd.read_csv(filepath, nrows=0).columns.tolist()
    with open(filepath, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1
    return render_template('results.html', file_id=file_id, columns=columns, total_rows=total_rows)


@app.route('/get_page', methods=['GET'])
def get_page():
    file_id = session.get('file_id')
    if not file_id: return jsonify({"error": "No active session"}), 403
    page     = int(request.args.get('page', 1))
    size     = int(request.args.get('size', 15))
    filepath = os.path.join(TEMP_DIR, f"{file_id}.csv")
    if not os.path.exists(filepath): return jsonify({"error": "File not found"}), 404
    cols      = pd.read_csv(filepath, nrows=0).columns.tolist()
    skip_rows = (page - 1) * size + 1
    try:
        chunk = pd.read_csv(filepath, skiprows=skip_rows, nrows=size, header=None, names=cols)
        chunk = chunk.replace({np.nan: None})
        return jsonify(chunk.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/download_csv', methods=['GET'])
def download_csv():
    file_id = session.get('file_id')
    if not file_id: return redirect('/')
    filepath = os.path.join(TEMP_DIR, f"{file_id}.csv")
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name='fraud_predictions.csv')
    return "File not found", 404


@app.route('/stats', methods=['GET'])
def stats_page():
    file_id = session.get('file_id')
    if not file_id: return redirect('/')
    if not os.path.exists(os.path.join(TEMP_DIR, f"{file_id}.csv")): return redirect('/')
    return render_template('stats.html', file_id=file_id)


@app.route('/api/stats_data', methods=['GET'])
def get_stats_data():
    file_id = session.get('file_id')
    if not file_id: return jsonify({"error": "No active session"}), 403
    filepath = os.path.join(TEMP_DIR, f"{file_id}.csv")
    if not os.path.exists(filepath): return jsonify({"error": "File not found"}), 404

    try:
        df          = pd.read_csv(filepath)
        total_tx    = len(df)
        total_fraud = int(df['fraud_prediction'].sum())
        fraud_pct   = round((total_fraud / total_tx) * 100, 2) if total_tx > 0 else 0.0

        if 'transaction_id' not in df.columns:
            df['transaction_id'] = ["TXN_" + str(i) for i in df.index]

        # Top 5 suspicious
        top_sus  = df.nlargest(5, 'fraud_probability')
        top_list = top_sus[['transaction_id', 'fraud_probability']].to_dict(orient='records')

        # Distribution bins
        bins   = [0, 20, 40, 60, 80, 100]
        labels = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%']
        df['prob_bin'] = pd.cut(df['fraud_probability'], bins=bins, labels=labels, include_lowest=True)
        dist_counts    = df['prob_bin'].value_counts().reindex(labels).fillna(0).astype(int).tolist()

        # High-risk count (>=80%)
        high_risk_count = int((df['fraud_probability'] >= 80).sum())

        # --- Threshold sensitivity: how many % are flagged at thresholds 10,20...90 ---
        thresholds = list(range(10, 100, 10))
        thresh_sensitivity = []
        for t in thresholds:
            flagged_pct = round(float((df['fraud_probability'] >= t).sum() / total_tx * 100), 2)
            thresh_sensitivity.append({"threshold": t, "flagged_pct": flagged_pct})

        # --- Average fraud score per probability band ---
        band_avg_scores = []
        for lbl in labels:
            subset = df[df['prob_bin'] == lbl]['fraud_probability']
            avg    = round(float(subset.mean()), 2) if len(subset) > 0 else 0.0
            band_avg_scores.append(avg)

        return jsonify({
            "total"              : total_tx,
            "frauds"             : total_fraud,
            "fraud_pct"          : fraud_pct,
            "top_suspicious"     : top_list,
            "dist_labels"        : labels,
            "dist_counts"        : dist_counts,
            "high_risk_count"    : high_risk_count,
            "thresh_sensitivity" : thresh_sensitivity,
            "band_avg_scores"    : band_avg_scores,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)