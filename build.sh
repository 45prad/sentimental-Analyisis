#!/usr/bin/env bash
# Exit immediately on error
set -o errexit

# Upgrade pip to latest version
pip install --upgrade pip

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Download required NLTK data (stopwords)
python -c "import nltk; nltk.download('stopwords')"

# (Optional) Add any other build steps needed for your app
# Example:
# - Database migrations: `flask db upgrade`
# - Collect static files: `flask collectstatic --noinput`