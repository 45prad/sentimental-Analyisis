#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords (required for your app)
python -c "import nltk; nltk.download('stopwords')"

# Collect static files (if any)
python manage.py collectstatic --no-input