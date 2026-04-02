# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for efficient caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY functions.py .
COPY template/ ./template/

# Create temp directory for predictions
RUN mkdir -p temp_predictions

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run Flask app with gunicorn (single worker for memory efficiency)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--worker-class", "sync", "--timeout", "120", "app:app"]
