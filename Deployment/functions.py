import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

REPO_ID = 'mlwithprince/Fruad-Deteaction_models'

def normalize_browser(browser):
    if pd.isna(browser): return "Other"
    s = browser.lower()
    if "chrome" in s or "chromium" in s: return "Chrome"
    if "safari" in s: return "Safari"
    if "firefox" in s: return "Firefox"
    if "edge" in s: return "Edge"
    if "ie" in s or "internet explorer" in s: return "IE"
    if "samsung" in s: return "Samsung"
    if "opera" in s: return "Opera"
    if "webview" in s or "android browser" in s: return "Android WebView"
    return "Other"

def normalize_os(os_str):
    if pd.isna(os_str): return "Other"
    s = os_str.lower()
    if "windows" in s: return "Windows"
    if "ios" in s: return "iOS"
    if "android" in s: return "Android"
    if "mac" in s: return "macOS"
    if "linux" in s: return "Linux"
    return "Other"

def make_env_risk(env_freq):
    if env_freq < 0.005: return 2
    if env_freq < 0.05:  return 1
    return 0

def eng_feas(df):
    df["user_browser"] = df["user_browser"].apply(normalize_browser)
    df["user_os"] = df["user_os"].apply(normalize_os)

    df['card_info'] = df['card_network'] + '_' + df['card_type']
    df['card_type_user_os'] = df['card_type'] + '_' + df['user_os']

    df.sort_values(by="transaction_time", inplace=True)
    df['time_hour'] = (df['transaction_time'] // 3600) % 24
    df['time_diff'] = df['transaction_time'].diff()
    df['new_day'] = df['time_hour'].diff() < 0
    df['time_day'] = df['transaction_time'] / 86400
    df['day_of_week'] = df['time_day'].astype(int) % 7

    df['time_diff'] = df['time_diff'].fillna(0)
    df['transaction_amount'] = df['transaction_amount'].fillna(0)

    df['time_diff_log'] = np.log1p(df['time_diff'])
    df['amt_log'] = np.log1p(df['transaction_amount'])

    try:
        df['amt_bins'] = pd.qcut(df['amt_log'], q=10, labels=False, duplicates='drop')
    except:
        df['amt_bins'] = pd.cut(df['amt_log'], bins=10, labels=False)

    try:
        df['time_diff_bins'] = pd.qcut(df['time_diff_log'], q=20, labels=False, duplicates='drop')
    except:
        df['time_diff_bins'] = pd.cut(df['time_diff_log'], bins=20, labels=False)

    df['environment'] = (
        df['device_type'].fillna('missing') + '_' +
        df['user_os'].fillna('missing') + '_' +
        df['user_browser'].fillna('missing')
    )
    env_freq = df['environment'].value_counts(normalize=True)
    df['environment_freq'] = df['environment'].map(env_freq)
    df['environment_risk'] = df['environment_freq'].apply(make_env_risk)

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(0)
    df = df.fillna('nan')

    try:
        dtype_path = hf_hub_download(repo_id=REPO_ID, filename='train_dtypes.pkl')
        train_dtypes = joblib.load(dtype_path)
        for col in df.columns:
            if col in train_dtypes:
                try:
                    df[col] = df[col].astype(train_dtypes[col])
                except:
                    pass
    except:
        pass

    return df

def select_features(df):
    final_cols = [
        'transaction_id', 'card_type', 'purchaser_email_domain', 'device_type',
        'is_identity_seen_before', 'user_os', 'user_browser', 'environment_risk',
        'card_info', 'time_hour', 'amt_bins', 'day_of_week', 'time_diff_log',
        'amt_log', 'time_diff_bins', 'card_type_user_os', 'card_network'
    ]
    existing = [c for c in final_cols if c in df.columns]
    return df[existing]

def process_csv(file):
    required = [
        'transaction_id', 'user_browser', 'user_os', 'card_type',
        'transaction_time', 'transaction_amount', 'device_type'
    ]
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.replace(r'^\ufeff', '', regex=True)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = eng_feas(df)
    df = select_features(df)
    return df