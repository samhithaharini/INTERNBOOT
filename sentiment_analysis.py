
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
import json
from datetime import datetime

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_and_preprocess_dataset(path):
    print("Loading and preprocessing Sentiment140 dataset...")
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(path, names=columns, encoding='latin-1')
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    print("Preprocessing complete. Sample entry:")
    print(df[['text', 'processed_text']].head(1))
    return df

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def vader_sentiment(self, text):
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive', compound
        elif compound <= -0.05:
            return 'negative', compound
        else:
            return 'neutral', compound
    
    def textblob_sentiment(self, text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    
    def combined_sentiment(self, text):
        vader_result, vader_score = self.vader_sentiment(text)
        textblob_result, textblob_score = self.textblob_sentiment(text)
        final_score = (vader_score * 0.6) + (textblob_score * 0.4)
        if final_score > 0.1:
            return 'positive', final_score
        elif final_score < -0.1:
            return 'negative', final_score
        else:
            return 'neutral', final_score


class ConversationLogger:
    def __init__(self, log_file="conversation_logs.json"):
        self.log_file = log_file
    
    def log(self, user_message, bot_response, sentiment, score):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "sentiment": sentiment,
            "confidence_score": score
        }
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
        logs.append(entry)
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)


class SentimentChatbot:
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.logger = ConversationLogger()
    
    def get_response(self, user_input):
        sentiment, score = self.analyzer.combined_sentiment(user_input)
        
        if sentiment == 'positive':
            response = f"😊 I'm happy that you're feeling positive! (confidence: {score:.2f})"
        elif sentiment == 'negative':
            response = f"😔 I'm sorry you're feeling that way. (confidence: {abs(score):.2f})"
        else:
            response = f"😐 I see, thanks for sharing your thoughts. (neutral tone detected)"
        
       
        self.logger.log(user_input, response, sentiment, score)
        return response

if __name__ == "__main__":
    print("Launching AI Chatbot with Sentiment Analysis...")
    print("Type 'exit' to stop the chat.\n")

  
    chatbot = SentimentChatbot()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: 👋 Goodbye! Have a nice day.")
            break
        bot_response = chatbot.get_response(user_input)
        print(f"Bot: {bot_response}")
