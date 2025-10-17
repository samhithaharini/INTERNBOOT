import spacy
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import nltk



nlp = spacy.load("en_core_web_sm")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def extract_keywords(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

data = pd.read_csv("dataset.csv")

data['symptoms'] = data['symptoms'].apply(normalize_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['symptoms'])
y = data['disease']
clf = RandomForestClassifier()
clf.fit(X, y)

def predict_with_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    found_symptoms = [ent[0].lower() for ent in entities if ent[1] in ("SYMPTOM", "DISEASE")]
    found_symptoms.extend(extract_keywords(text))
    found_symptoms = list(set(found_symptoms))
    return found_symptoms, entities

def predict_with_ml(text):
    normalized = normalize_text(text)
    features = vectorizer.transform([normalized])
    return clf.predict(features)[0]

app = Flask(__name__)

@app.route('/check', methods=['POST'])
def check_symptoms():
    user_input = request.json.get("text", "")
    found_symptoms, entities = predict_with_ner(user_input)
    ml_pred = predict_with_ml(user_input)
    return jsonify({
        "input": user_input,
        "extracted_symptoms": found_symptoms,
        "named_entities": entities,
        "ml_prediction": ml_pred
    })


if __name__ == "__main__":
    user_text = input("Describe your symptoms: ")
    found_symptoms, entities = predict_with_ner(user_text)
    print(f"\nNamed Entities: {entities}")
    print(f"Extracted Symptoms: {found_symptoms}")
    print(f"ML Baseline Prediction: {predict_with_ml(user_text)}")

