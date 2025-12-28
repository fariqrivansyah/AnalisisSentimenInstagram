from flask import Flask, render_template, request
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# NLP tools
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        text = request.form["comment"]
        clean_text = preprocess(text)
        vector = tfidf.transform([clean_text])
        result = model.predict(vector)[0]
        prediction = "Positif 😊" if result == 1 else "Negatif 😡"

    return render_template("index.html", prediction=prediction)
if __name__ == "__main__":
    app.run(debug=True)