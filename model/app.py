#!/usr/bin/env python3

import psycopg2
import pandas as pd
import nltk
import re
import nltk
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

lemm_df['scores']=lemm_df['Lemmatized_text'].apply(lambda Lemmatized_text :sid.polarity_scores(Lemmatized_text))
lemm_df['compound']=lemm_df['scores'].apply(lambda score_dict:score_dict['compound'])

lemm_df['comp_score']=lemm_df['compound'].apply(lambda c:'pos' if c>=0 else "neg")

lemm_df.head()

lemm_df['comp_score'].value_counts()

lemm_df=lemm_df.iloc[:20000]
lemm_df.shape

cv=CountVectorizer()

X=cv.fit_transform(lemm_df['Lemmatized_text']).toarray()
Y=pd.get_dummies(lemm_df['comp_score'])
Y=Y.iloc[:,1].values


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
model=MultinomialNB().fit(x_train,y_train)
y_pred=model.predict(x_test)

print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
svm = sn.heatmap(cm, annot=True, fmt = '.2f')
figure = svm.get_figure()    
figure.savefig('model/svm_conf.png', dpi=400)

dump(cv,open('model/vectorizer.pkl','wb'))

dump(model,open('model/model.pkl','wb'))
