#!/usr/bin/env python3

import psycopg2
import pandas as pd
import nltk
import re
import nltk
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, SimpleRNN, advanced_activations
from keras.wrappers.scikit_learn import KerasClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pickle import dump
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sn

ps = PorterStemmer()
ls = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))


connection = psycopg2.connect(user="twitt_producer",
                              password="Loh6ziet",
                              host="postgresql",
                              port="5432",
                              database="sentimental_analysis")
cursor = connection.cursor()
cursor2 = connection.cursor()
path='model/tweets_pg_export.csv'

sql = "COPY (SELECT username,text FROM twitts) TO STDOUT WITH CSV HEADER DELIMITER ';'"
with open(path, "w") as file:
    cursor.copy_expert(sql, file)

cursor.close()
connection.close()

df = pd.read_csv(path, sep=';')              
print(df.shape)
tweets = df['text']
print(tweets)

def rnn():
  model_rnn = Sequential()

  model_rnn.add(Embedding(input_dim=30, output_dim=30))
  
  model_rnn.add(Masking(mask_value=0.0))

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def cleaning(text):
    text = decontracted(text)
    text = text.lower()                              #lowering the text
    text = re.sub(r'@\S+','',text)                   #Removed @mentions
    text = re.sub(r'#\S+','',text)                   #Remove the hyper link
    text = re.sub(r'RT\S+','',text)                  #Removing ReTweets
    text = re.sub(r'https?\S+','',text)              #Remove the hyper link
    text = re.sub('[^a-z]',' ',text)              #Remove the character other than alphabet
    text = text.split()
    return text
tweets_cleaned =  tweets.apply(cleaning)
tweets_cleaned

stop_words = set(stopwords.words('english'))
tweets_stemmed = tweets_cleaned.apply(lambda x: [ps.stem(word) for word in x if word not in stop_words])
tweets_lemmatized = tweets_cleaned.apply(lambda x: [ls.lemmatize(word) for word in x if word not in stop_words])


tweets_stemmed = tweets_stemmed.apply(lambda x: ' '.join(x))
tweets_lemmatized = tweets_lemmatized.apply(lambda x: ' '.join(x))


df['scrapped_text'] = df['text']
df['Lemmatized_text'] = tweets_lemmatized.to_frame() 
df['Stemmed_text'] = tweets_stemmed.to_frame()
new_df=df.drop(['username',],axis=1)

new_df.head()

new_df.isnull().sum()
df=new_df.dropna(axis=0)

df.shape

stem_df=df[['Stemmed_text']]
lemm_df=df[['Lemmatized_text']]

nltk.download('vader_lexicon')
sid=SentimentIntensityAnalyzer()

stem_df['scores']=stem_df['Stemmed_text'].apply(lambda Stemmed_text :sid.polarity_scores(Stemmed_text))
stem_df['compound']=stem_df['scores'].apply(lambda score_dict:score_dict['compound'])

stem_df['comp_score']=stem_df['compound'].apply(lambda c:'pos' if c>=0 else "neg")

stem_df.head()

stem_df['comp_score'].value_counts()

stem_df=stem_df.iloc[:20000]
stem_df.shape

cv=CountVectorizer()

X=cv.fit_transform(stem_df['Stemmed_text']).toarray()
Y=pd.get_dummies(stem_df['comp_score'])
Y=Y.iloc[:,1].values



x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

model = Sequential()
model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())

batch_size=32
model.fit(X_train, y_train, epochs = 5, batch_size=batch_size, verbose = 'auto')
model.evaluate(X_test,y_test)

print("Prediction: ",model.predict_classes(X_test[5:10]))

print("Actual: \n",y_test[5:10])


dump(cv,open('model/vectorizer.pkl','wb'))

dump(model,open('model/model.pkl','wb'))
