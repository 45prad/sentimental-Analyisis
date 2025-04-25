# # Library imports
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import re
# from numpy import array




# from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Flatten, GlobalMaxPooling1D, Embedding, Conv1D
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import one_hot, Tokenizer, tokenizer_from_json


# # from keras.models import Sequential, load_model
# # from keras.layers import LSTM
# # from keras.layers.core import Activation, Dropout, Dense
# # from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
# from sklearn.model_selection import train_test_split
# from flask import Flask, request, jsonify, render_template
# # from keras_preprocessing.sequence import pad_sequences
# import nltk
# from nltk.corpus import stopwords
# # from keras_preprocessing.text import tokenizer_from_json
# import io
# import json
# nltk.download('stopwords')
# stopwords_list = set(stopwords.words('english'))
# maxlen = 100

# # Load model
# model_path ='c1_lstm_model_acc_0.861.h5'
# pretrained_lstm_model = load_model(model_path)

# # Loading
# with open('b3_tokenizer.json') as f:
#     data = json.load(f)
#     loaded_tokenizer = tokenizer_from_json(data)


# # Create the app object
# app = Flask(__name__)


# # creating function for data cleaning
# from b2_preprocessing_function import CustomPreprocess
# custom = CustomPreprocess()


# # Define predict function
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     query_asis = [str(x) for x in request.form.values()]
# #     query_list = []
# #     query_list.append(query_asis)
    
#     # Preprocess review text with earlier defined preprocess_text function
#     query_processed_list = []
#     for query in query_asis:
#         query_processed = custom.preprocess_text(query)
#         query_processed_list.append(query_processed)
        
#     # Tokenising instance with earlier trained tokeniser
#     query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)
    
#     # Pooling instance to have maxlength of 100 tokens
#     query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)
    
#     # Passing tokenised instance to the LSTM model for predictions
#     query_sentiments = pretrained_lstm_model.predict(query_padded)
    

#     if query_sentiments[0][0]>0.5:
#         return render_template('index.html', prediction_text=f"Positive Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")
#     else:
#         return render_template('index.html', prediction_text=f"Negative Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")


# if __name__ == "__main__":
#     app.run(debug=True)


# Library imports
from flask import Flask, request, jsonify
from flask_cors import CORS  # For cross-origin requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
maxlen = 100
nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))

# Load model and tokenizer
try:
    model_path = 'c1_lstm_model_acc_0.861.h5'
    pretrained_lstm_model = load_model(model_path)
    
    with open('b3_tokenizer.json') as f:
        tokenizer_data = json.load(f)
        loaded_tokenizer = tokenizer_from_json(tokenizer_data)
except Exception as e:
    print(f"Error loading model/tokenizer: {str(e)}")
    exit(1)

# Import your custom preprocessing
try:
    from b2_preprocessing_function import CustomPreprocess
    custom = CustomPreprocess()
except ImportError:
    print("Error importing custom preprocessing module")
    exit(1)

@app.route('/analyze_feedback', methods=['POST', 'OPTIONS'])
def analyze_feedback():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({'status': 'ok'}), 200
    
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({"error": "Invalid data format. Expected a list of events."}), 400
        
        results = []
        
        for event in data:
            # Initialize counters
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            feedbacks = event.get('Feedbacks', [])
            analyzed_feedbacks = []
            
            for fb in feedbacks:
                comment = fb.get('comments', '')
                if not comment:
                    neutral_count += 1
                    analyzed_feedbacks.append({
                        **fb,
                        "sentiment": "neutral",
                        "rating": 0
                    })
                    continue
                
                try:
                    # Preprocess comment
                    processed_comment = custom.preprocess_text(comment)
                    
                    # Tokenize and pad
                    tokenized = loaded_tokenizer.texts_to_sequences([processed_comment])
                    padded = pad_sequences(tokenized, padding='post', maxlen=maxlen)
                    
                    # Predict sentiment
                    sentiment_score = pretrained_lstm_model.predict(padded, verbose=0)[0][0]
                    rating = float(np.round(sentiment_score * 10, 1))
                    
                    # Classify sentiment
                    if sentiment_score > 0.6:
                        sentiment = "positive"
                        positive_count += 1
                    elif sentiment_score < 0.4:
                        sentiment = "negative"
                        negative_count += 1
                    else:
                        sentiment = "neutral"
                        neutral_count += 1
                    
                    analyzed_feedbacks.append({
                        **fb,
                        "sentiment": sentiment,
                        "rating": rating
                    })
                except Exception as e:
                    print(f"Error processing comment: {str(e)}")
                    neutral_count += 1
                    analyzed_feedbacks.append({
                        **fb,
                        "sentiment": "neutral",
                        "rating": 0
                    })
            
            results.append({
                "eventId": event.get('eventId'),
                "eventName": event.get('eventName'),
                "totalFeedback": len(feedbacks),
                "positiveCount": positive_count,
                "negativeCount": negative_count,
                "neutralCount": neutral_count,
                "Feedbacks": analyzed_feedbacks
            })
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=True)