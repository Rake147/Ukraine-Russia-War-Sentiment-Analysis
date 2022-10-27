#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string


# In[2]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/filename.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data=data[["username","tweet","language"]]


# In[6]:


data


# In[7]:


data.isnull().sum()


# In[8]:


data['language'].value_counts()


# In[9]:


nltk.download('stopwords')
stemmer=nltk.SnowballStemmer('english')
stopword=set(stopwords.words('english'))


# In[10]:


def clean(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]','', text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=[word for word in text.split(' ') if word not in stopword]
    text=' '.join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=' '.join(text)
    return text


# In[11]:


data['tweet']=data['tweet'].apply(clean)


# In[12]:


text = ' '.join(i for i in data.tweet)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[13]:


nltk.download('vader_lexicon')
sentiments=SentimentIntensityAnalyzer()
data['Positive']=[sentiments.polarity_scores(i)['pos'] for i in data['tweet']]
data['Negative']=[sentiments.polarity_scores(i)['neg'] for i in data['tweet']]
data['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['tweet']]
data=data[["tweet","Positive","Negative","Neutral"]]
print(data.head())


# In[14]:


# Most frequent words used 
positive =' '.join([i for i in data['tweet'][data['Positive'] > data['Negative']]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color='white').generate(positive)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[17]:


# NEgative Sentiments
negative=' '.join([i for i in data['tweet'][data['Negative']>data['Positive']]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords,background_color='white').generate(negative)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:




