import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def normalize_browser(browser):
    if pd.isna(browser):
        return "Other"
    s = browser.lower()
    if "chrome" in s or "chromium" in s:
        return "Chrome"
    elif "safari" in s:
        return "Safari"
    elif "firefox" in s:
        return "Firefox"
    elif "edge" in s:
        return "Edge"
    elif "ie" in s or "internet explorer" in s:
        return "IE"
    elif "samsung" in s:
        return "Samsung"
    elif "opera" in s:
        return "Opera"
    elif "webview" in s or "android browser" in s:
        return "Android WebView"
    else:
        return "Other"

def normalize_os(os_str):
    if pd.isna(os_str):
        return "Other"
    s = os_str.lower()
    if "windows" in s:
        return "Windows"
    elif "ios" in s:
        return "iOS"
    elif "android" in s:
        return "Android"
    elif "mac" in s:
        return "macOS"
    elif "linux" in s:
        return "Linux"
    else:
        return "Other"

def make_env_risk(env_freq):
    if env_freq < 0.005:
        return 2      # very rare
    elif env_freq < 0.05:
        return 1      # rare
    else:
        return 0      # common

def ensure_numeric(series):
    """Convert to numeric, coercing errors to NaN"""
    return pd.to_numeric(series, errors='coerce')

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def eng_feas(df):
    # Normalize browser and OS
    df["user_browser"] = df["user_browser"].apply(normalize_browser)
    df["user_os"] = df["user_os"].apply(normalize_os)

    # Fill missing values with 'missing' (instead of 'nan')
    df = df.fillna('nan')

    # Create interaction features
    df['card_info'] = df['card_network'] + '_' + df['card_type']
    df['card_type_user_os'] = df['card_type'] + '_' + df['user_os']

    # Time-based features
    df.sort_values(by="transaction_time", inplace=True)
    df['time_hour'] = (df['transaction_time'] // 3600) % 24
    df['time_diff'] = df['transaction_time'].diff()
    df['new_day'] = df['time_hour'].diff() < 0
    df['time_day'] = df['transaction_time'] / 86400
    df['day_of_week'] = df['time_day'].astype(int) % 7

    # Log transformations
    df['time_diff_log'] = np.log1p(ensure_numeric(df['time_diff']))
    df['amt_log'] = np.log1p(ensure_numeric(df['transaction_amount']))

    # Binning
    try:
        df['amt_bins'] = pd.qcut(df['amt_log'], q=10, labels=False, duplicates='drop')
    except ValueError:
        # Fallback if quantiles fail (e.g., too few unique values)
        df['amt_bins'] = pd.cut(df['amt_log'], bins=10, labels=False)

    try:
        df['time_diff_bins'] = pd.qcut(df['time_diff_log'], q=20, labels=False, duplicates='drop')
    except ValueError:
        df['time_diff_bins'] = pd.cut(df['time_diff_log'], bins=20, labels=False)

    # Environment risk
    df['environment'] = (
        df['device_type'].fillna('missing') + '_' +
        df['user_os'].fillna('missing') + '_' +
        df['user_browser'].fillna('missing')
    )
    env_freq = df['environment'].value_counts(normalize=True)
    df['environment_freq'] = df['environment'].map(env_freq)
    df['environment_risk'] = df['environment_freq'].apply(make_env_risk)

    # Convert dtypes using saved template (if available)
    df = change_dtypes(df)

    return df

def change_dtypes(df):
    """Convert columns to the dtypes saved during training."""
    try:
        train_dtypes = joblib.load('/media/prince/5A4E832F4E83034D/Fraud-Detector-ML/Models/Final/train_dtypes.pkl')
        # Only convert columns that exist in both the DataFrame and the saved dtypes
        for col in df.columns:
            if col in train_dtypes:
                try:
                    df[col] = df[col].astype(train_dtypes[col])
                except (TypeError, ValueError):
                    # If conversion fails, keep as is
                    pass
    except FileNotFoundError:
        print("Warning: train_dtypes.pkl not found. Skipping dtype conversion.")
    return df

def select_features(df):
    """Keep only the finalized features (including transaction_id)."""
    finalized_features = [
        'transaction_id',          
        'card_type',
        'purchaser_email_domain',
        'device_type',
        'is_identity_seen_before',
        'user_os',
        'user_browser',
        'environment_risk',
        'card_info',
        'time_hour',
        'amt_bins',
        'day_of_week',
        'time_diff_log',
        'amt_log',
        'time_diff_bins',
        'card_type_user_os',
        'card_network'
    ]
    # Keep only columns that exist in the DataFrame
    available = [col for col in finalized_features if col in df.columns]
    return df[available]

# ----------------------------------------------------------------------
# Main data processing function (for Flask)
# ----------------------------------------------------------------------
def data_check(file):
    """
    Process uploaded CSV: clean, engineer features, select final columns, save.
    Returns a dict with status and metadata.
    """
    required_columns = [
        'transaction_id', 'user_browser', 'user_os', 'card_type',
        'transaction_time', 'transaction_amount', 'device_type'
    ]

    try:
        # Read CSV
        df = pd.read_csv(file)

        # Clean column names: remove BOM, extra spaces
        df.columns = df.columns.str.strip().str.replace(r'^\ufeff', '', regex=True)

        # Verify required columns exist
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Backup transaction_id (though it's kept later)
        # No need to copy separately; it stays in df

        # Apply feature engineering
        df = eng_feas(df)

        # Select final features
        df = select_features(df)

        # Save processed data
        output_path = '/media/prince/5A4E832F4E83034D/Fraud-Detector-ML/Data/processed_data.csv'
        df.to_csv(output_path, index=False)

        return {
            'status': 'success',
            'message': f'Processed {len(df)} rows. Saved to {output_path}',
            'rows_processed': len(df),
            'features_used': list(df.columns),
            'saved_path': output_path
        }

    except Exception as e:
        # Log the error (you might want to use proper logging)
        print("Error during data_check:", str(e))
        if 'df' in locals():
            print("Columns in dataframe:", df.columns.tolist())
        raise Exception(f"Data check failed: {str(e)}")