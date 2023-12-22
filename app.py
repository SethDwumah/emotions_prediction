import re
import pickle
import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# from xgboost import XGBClassifier


app = Flask(__name__)


def preprocess_text(text):
    sentence = re.sub(r'[^\w\s]', ' ', text)
    word = [' '.join(token.lower() for token in nltk.word_tokenize(sentence)
                    if token not in stopwords.words('english'))]
    return word

def load_model():
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

Vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    feelings = request.form['feelings']
    # feelings_transformed = preprocess_text(feelings)
    text_array = np.array([feelings])
    feelings_vectorized = Vectorizer.transform(text_array)
    model = load_model()
    prediction = model.predict(feelings_vectorized)
    return render_template('results1.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)