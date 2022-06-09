import re
import nltk

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from lib.logger import Logger 
from model.tweet import Tweet



class TwitterSentimentalProcessor(object):
    
    def __init__(self, database):
        nltk.download('stopwords')
        self._ps = PorterStemmer()
        self._vectorizer = load(open('ml/countvectorizer.pkl','rb'))
        self._classifier = load(open('ml/model.pkl','rb'))

        self._db = database

    
    def analyze(self, payload):
        payload = self._clean(payload)
        payload_encoded = self._vectorizer.transform(payload)
        payload_encoded = payload_encoded.toarray()

        return "Positive" if self._classifier.predict(payload_encoded)  == 1 else "Negative"

    def _clean(self, payload):
        corpus = []
        payload = self._decontracte(payload)
        payload = payload.lower()                              #lowering the payload
        payload = re.sub(r'@\S+','',payload)                   #Removed @mentions
        payload = re.sub(r'#\S+','',payload)                   #Remove the hyper link
        payload = re.sub(r'RT\S+','',payload)                  #Removing ReTweets
        payload = re.sub(r'https?\S+','',payload)              #Remove the hyper link
        payload = re.sub('[^a-z]',' ',payload)              #Remove the character other than alphabet
        payload = payload.split()
        payload=[self._ps.stem(word) for word in payload if word not in stopwords.words('english')]
        payload=' '.join(payload)
        corpus.append(payload)
        
        return corpus

    def _decontracte(self, payload):
        # specific
        payload = re.sub(r"won\'t", "will not", payload)
        payload = re.sub(r"can\'t", "can not", payload)

        # general
        payload = re.sub(r"n\'t", " not", payload)
        payload = re.sub(r"\'re", " are", payload)
        payload = re.sub(r"\'s", " is", payload)
        payload = re.sub(r"\'d", " would", payload)
        payload = re.sub(r"\'ll", " will", payload)
        payload = re.sub(r"\'t", " not", payload)
        payload = re.sub(r"\'ve", " have", payload)
        payload = re.sub(r"\'m", " am", payload)
        
        return payload

    def callback(self, ch, method, properties, body):
        payload = Tweet().from_json(str(body,'utf-8'))
        sentiment = self.analyze(payload.get_text())

        self._db.insert_twitt(payload.get_username(),
                     payload.get_text(),
                     payload.get_created_at(),
                     sentiment
                    )

        Logger.info(f"Text: {payload.get_text()} Result: {sentiment}")