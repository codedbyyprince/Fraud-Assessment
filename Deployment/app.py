from flask import Flask, render_template, request
from functions import data_check
import pandas as pd

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'fileToUpload' not in request.files:
        return "No file provided", 400
    
    file = request.files['fileToUpload']
    
    # Check if CSV
    if not file.filename.endswith('.csv'):
        return "Error: File must be CSV format", 400
    
    # Data check
    try:
        data_check(file)
        return "Data checked and saved successfully", 200
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)