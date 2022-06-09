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
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

ps = PorterStemmer()
ls = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df = pd.read_csv("coronavirus.csv")              
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
df['Stemmed_text'] = tweets_lemmatized.to_frame()

new_df=df.drop(['Unnamed: 0', 'id', 'username',  'created_at','text',],axis=1)

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
model=MultinomialNB().fit(x_train,y_train)
y_pred=model.predict(x_test)

print(classification_report(y_test,y_pred))
cm = metrics.confusion_matrix(y_test, y_pred)
sn.heatmap(cm, annot=True, fmt = '.2f')


dump(cv,open('pickle/vectorizer.pkl','wb'))

dump(model,open('pickle/model.pkl','wb'))
