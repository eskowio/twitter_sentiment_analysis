from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, SimpleRNN, advanced_activations, BatchNormalization, Flatten, MaxPooling1D, SpatialDropout1D, MaxPooling2D, Conv1D,  Conv2D, Flatten, LeakyReLU, Reshape
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.model_selection  import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn import metrics, preprocessing
import seaborn as sn
import re
import nltk
from keras import layers
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import time
from google.colab import drive
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
ps = PorterStemmer()
ls = WordNetLemmatizer()
drive.mount('/content/drive')
df = pd.read_csv("drive/MyDrive/coronavirus.csv")              
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
stop_words = set(stopwords.words('english'))
def cleaning(text):
    text = decontracted(text)
    text = text.lower()                             
    text = re.sub(r'@\S+','',text)                   
    text = re.sub(r'#\S+','',text)                   
    text = re.sub(r'RT\S+','',text)                  
    text = re.sub(r'https?\S+','',text)             
    text = re.sub('[^a-z]',' ',text)             
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
new_df=df.drop(['Unnamed: 0', 'id', 'permalink', 'username', 'to', 'text', 'created_at','retweets', 'favorites', 'mentions', 'hashtags', 'geo',],axis=1)
new_df.head()

df = new_df
df.head()
df.shape
df.isnull().sum()
df=df.dropna(axis=0)
df.shape

stem_df=df[['Stemmed_text']]
lemm_df=df[['Lemmatized_text']]
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
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
Y
X.shape,Y.shape

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=1)

x_train
x_test
y_train
y_test
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
def modelSentiment():
  # model 2 
  #model = Sequential()
  #model.add(Embedding(100, 100, input_length=X.shape[1]))
  #model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
  #model.add(MaxPooling1D(pool_size=2))
  #model.add(Dropout(0.2))
  #model.add(Flatten())
  #model.add(Dense(12, activation='relu'))
  #model.add(Dropout(0.4))
  #model.add(Dense(1, activation='sigmoid'))# compile network
  #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  #print(model.summary())
  #return model

  # model 3
  model = Sequential()
  model.add(Embedding(100, 100, input_length=X.shape[1]))
  model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(12, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(1, activation='sigmoid'))# compile network
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  return model

model8 = KerasClassifier(build_fn=modelSentiment,epochs=20)
start = time.time()
model8.fit(x_train, y_train)
end = time.time()

print('Training time')
print((end-start))
history = model8.fit(x_train,
                    y_train,
                      epochs=10,
                      batch_size=256,
                      validation_data=(x_test, y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation accuracy'], loc='upper left') 
plt.show()
plt.savefig('Model_accuracy.pdf')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation loss'], loc='lower left')  
plt.show()
plt.savefig('Model_loss.pdf')

y_pred_val = model8.predict(x_test)


print(classification_report(y_test,y_pred_val))
cm = metrics.confusion_matrix(y_test, y_pred_val)
sn.heatmap(cm, annot=True, fmt = '.2f')
