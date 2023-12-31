import numpy as np
import pickle
import streamlit as st
import nltk
nltk.download('stopwords')
import sklearn
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

current_path = os.getcwd()
# print(current_path)
# import model
model_path = os.path.join(current_path, 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# import tfidf
tfidf_path = os.path.join(current_path, 'tfidf.pkl')
tfidf = pickle.load(open(tfidf_path, 'rb'))

# Try Testintg
st.title('Program Sistem Rekomendasi Minuman')
text = st.text_input("Tulis minuman seperti apa yang anda inginkan: ", placeholder="Contoh: Saya ingin minum kopi susu")

def predict(text):
    text = [text]
    text = tfidf.transform(text)

    predicted_probabilities = model.predict_proba(text)
    classes = model.classes_

    # Get top 5 classes
    top_5 = np.argsort(predicted_probabilities, axis=1)[:, -5:]
    top_5 = np.flip(top_5, axis=1)

    # Print top 5 classes
    st.subheader("Minuman rekomendasi untukmu:")
    for row in top_5:
        for index, idx in enumerate(row):
            # print(classes[idx], end=" ")
            # Remove [] and '
            result = re.sub(r"[\[\]']", "", classes[idx])
            st.write(index+1, result)
        # print()

#Remove indonesian Stopwords
def cleaning(text):
    stop_words = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    text = ' '.join([factory.create_stemmer().stem(word.lower()) for word in text.split()])
    text = ' '.join([word for word in text.split() if word not in (stop_words)])
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\d+', '')
    return text

if st.button('Recommend!'):
    predict(cleaning(text))

