#!/bin/bash
# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Start Gunicorn (adjust workers as needed)
gunicorn --workers 2 --bind 0.0.0.0:$PORT app:app