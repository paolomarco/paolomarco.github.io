---
layout: post
title: Analyzing the correlation between daily Bitcoin returns & crypto news using Python
---

### What is this about? 
The headline says it all. It started with a question: Is the price of bitcoin correlated to sentiment of news headlines about bitcoin and other crytocurrencies. 

Chances are you've heard about Bitcoin. Doesn't matter if you are into day trading or a HODLer for life, you'll have noticed that the price of Bitcoin is volatile. In 2017, the price of Bitcoin has risen by more than 1300% while YTD it's down -36%. Price swings of 15-20% are a fairly common occurence. 

If you follow along with the media, it's clear that just like the price of bitcoin, the news are a bit of a mixed bag. There are a lot of positive headlines talking about innovation, social impact, and positive response from goverments. There are also __A LOT__ of negative headlines about scam ICOs, bankrupt exchanges, sell-offs, and raids on illegal mining operations. 

Going back to the question: Are the two related? This post will attempt to answer that question. It will also cover how Python can be used to:
    * Scrape the web for Bitcoin news headlines
    * Train a sentiment classifier using the [Naive Bayes Algorithm](https://www.nltk.org/book/ch06.html) to determine sentiment
    * Ping a Bitcoin price API to calculate daily returns  
    * Calculate a correlation between daily returns & the % of daily crypto headlines classified as positive
    * Automate the analysis with [Launchd](http://www.launchd.info/)

### Are the two correlated?
In the absence of data, it would be logical to assume that the price of bitcoin would to some degree be affected by the sentiment of news headlines. In the presence of data, the statistical correlation between the daily price return & the % of daily crypto headlines classified as positive turns out to be _*0.52*_. While not a perfect linear relationship, this can be considered fairly strong. The below line chart shows this visually. The green line represents the % of daily headlines classified as positive (i.e. if 2 headlines appear on a given day & 1 is classfied as positive by the sentiment classifier, it would be a 0.5 for that day). The blue line represents daily returns calculated as: *(Today's price - Yesterday's price) / Yesterday's price*. 

![alt text](https://github.com/paolomarco/paolomarco.github.io/blob/master/images/linechart_sentiment_price.png?raw=true "Logo Title Text 1")

It would be interesting to do some further digging on if/how this could be used to make predictions about the price of future bitcoin. In addition, it would be interesting to explore what other variables could be included in a predictive model along with headline sentiment. 

### Approach
The diagram below shows the approach taken at a high level. The details and code to go along with each component of the flow chart are below. 

![alt text](https://github.com/paolomarco/paolomarco.github.io/blob/master/images/approach%202.jpg?raw=true "Logo Title Text 1")

#### _Headline web scraper_
The below code block was used to parse a major crypto news website & store the individual headlines in a local .csv file. It relies on an awesome Python package called BeautifulSoup which makes it easy to parse HTML documents & extract data. Using [Launchd](http://www.launchd.info/), this script was scheduled to run once a hour in the background (even when user is logged off from the computer). Depending on the HTML structure of the site, the code will have to be adjusted. 
```python
# import required libraries
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime as dt
import requests
import csv

# specify the url
quote_page = "www.some bitcoin news site.com" 
#  request page
page = requests.get(quote_page)
# parse html with beautifulsoup
soup = BeautifulSoup(page.content, 'html.parser')

# Create list to store headlines
headline_list = []

# parse feature articles
features = soup.find_all('div', class_="article article-featured")

for a in features:
    title = a.h3.text
    headline_list.append(title)

# parse sub feature articles
sub_features = soup.find_all('div', class_="post-info")
    
for a in sub_features:
    title = a.h3.text
    headline_list.append(title)
    
# add date & format as pandas data frame    
now = dt.datetime.now()
day = now.strftime("%Y-%m-%d")
x = np.array(day)
date_list = np.repeat(x, len(headline_list), axis=0)
df = pd.DataFrame({'title':np.array(headline_list), 'date':date_list})

headlines_list = df.values.tolist()

with open('~/headlines.csv') as exisiting_file: #insert correct directory
    existingLines = [line for line in csv.reader(exisiting_file, delimiter=',')]
    
for headline in headlines_list:
    if headline not in existingLines:
            existingLines.append(headline)
            
with open('~/headlines.csv', 'w') as f: #insert correct directory 
    pd.DataFrame(existingLines).to_csv(f,header=False,index=False)
```
#### _Training a Sentiment Classifier_
The next block of code involves training a supervised sentiment classifier (Naive Bayes), which can then be used to classify new incoming headlines. Supervised means that the model is trained using a dataset where both the input & outcome is known (headline text & correct positive/negative classification). The Naive Bayes classifier had an accuracy of 75% which was acceptable. The model was then stored as a file object using [pickle](https://docs.python.org/2/library/pickle.html). This is important because it will need to be re-loaded once the headline web scraper comes back with new headlines to be classified. 
```python
import nltk
from nltk.classify import NaiveBayesClassifier
import csv
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

csv_reader = csv.reader(open("~/train.csv", 'r'))
csv_list = list(csv_reader)

pos_train = []
neg_train = []

for i in csv_list:
    if i[1] == 'pos':
        pos_train.append([format_sentence(i[0]), 'pos'])
    else:
        neg_train.append([format_sentence(i[0]), 'neg'])

# next, split labeled data into the training and test data
training_2 = pos_train[:int((.8)*len(pos_train))] + neg_train[:int((.8)*len(neg_train))]
test_2 = pos_train[int((.8)*len(pos_train)):] + neg_train[int((.8)*len(neg_train)):]

# check if any items in pos_train exist in neg_train
for i in training_2:
    if i in test_2:
        print("duplicate")
 
 classifier_2 = NaiveBayesClassifier.train(training_2)
 #calculate accuracy
from nltk.classify.util import accuracy
print(accuracy(classifier_2, test_2))

pred = []
actual = []

for i in range(len(test_2)):
    pred.append(classifier_2.classify(test_2[i][0]))
    actual.append(test_2[i][1])

# storing model to disk
import pickle

file_Name = "~/nltk_classifier_2"
# open the file for writing
fileObject = open(file_Name,'wb')

# dump model to file
pickle.dump(classifier_2,fileObject) 

# Close the fileObject
fileObject.close()
```
#### _Classifying New Headlines_
Using pickle, the trained classifier can now reloaded & used to classify new incoming headlines. The code below does just that. Again, [Launchd](http://www.launchd.info/) was used to schedule the script to run once a day. 
```python
# import packages
import pickle
import nltk
import csv
import pandas

# import classifier
file_name = "~/nltk_classifier_2"
fileObject = open(file_name,'rb')
# load the object from the file into var classfier
classifier = pickle.load(fileObject)  
fileObject.close

# import data to classify & assign to var df
df = pandas.read_csv("~/headlines.csv", header = None)
df.columns = ["Date", "Headline"]

# def format sentence function required
def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

# create feature set
feature_set = []
for i in df["Headline"]:
    feature_set.append(format_sentence(i))
    
# classify feature set
classifications = []
for i in feature_set:
    classifications.append(classifier.classify(i))
    
# add classification column
df["classifications"] = classifications

# write to new csv onto desktop
df.to_csv("~/headlines_classified.csv", index = False)
```
#### _Calculating Bitcoin price returns_
The code below the API call to pull down Bitcoin daily prices from which returns can be calculated. Just like the headline classifier script, this was scheduled to run once a day.
```python
import requests
import json
import datetime as dt
import numpy as np
import pandas as pd
import csv
import locale
from collections import deque

# define functions required
def get_last_row(csv_filename):
    with open(csv_filename, 'r') as f:
        try:
            lastrow = deque(csv.reader(f), 1)[0]
        except IndexError:  # empty file
            lastrow = None
        return lastrow
    
# api call
url = 'www.api.something.com' # insert API here
resp = requests.get(url=url)
data = resp.json()

# format price + set current date
price_str = data['bpi']['USD']['rate']
price_float = float(price_str.replace(',',''))

now = dt.datetime.now()
day = now.strftime("%Y-%m-%d")

# put into list
list_to_append = [day, price_float]
todays_price = list_to_append[1]

last_row = get_last_row('~/bitcoin_daily_price.csv')

yesterdays_price = float(last_row[1].replace(',',''))

price_return = (todays_price - yesterdays_price)/yesterdays_price

list_to_append.append(price_return)

with open(r'~/bitcoin_daily_price.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(list_to_append)
```
#### Calculating the correlation between sentiment & price return
Now that both classified headlines & daily price returns are available, a correlation can be calculated. The % of daily headlines classified as positive is used as the sentiment proxy. The code below also generates the line chart above to show both variables visually. 
```python
# import dependencies
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

# prices
returns = pd.read_csv('~/bitcoin_daily_price.csv', header=None)
returns.columns = ['Date', 'Price','Return']
# format Date as datetime for merging later
returns['Date'] = pd.to_datetime(returns['Date'])
# classified headlines
sentiment = pd.read_csv('~/headlines_classified.csv', parse_dates=['Date'])

sentiment['count'] = [1] * len(sentiment.index)
sentiment_wide = sentiment.pivot_table(index = 'Date',columns = 'classifications', values = 'count',
                     aggfunc = 'sum')
#format
sentiment_wide = sentiment_wide.reset_index()
#replace NaNs w/ 0s
sentiment_wide = sentiment_wide.fillna(0)
#sum across colums to get total headline count
sentiment_wide['headline_sum'] = sentiment_wide.sum(axis=1)
#calculate positive percentage for the day
sentiment_wide['pos_perc'] = sentiment_wide['pos'] / sentiment_wide['headline_sum']
#calculate negative percentage for the day
sentiment_wide['neg_perc'] = sentiment_wide['neg'] / sentiment_wide['headline_sum']
merged_df = pd.merge(sentiment_wide, returns, on='Date')
merged_df['pos_perc'].corr(merged_df['Return'])
x = merged_df['Date']
y1 = merged_df['pos_perc']
y2 = merged_df['Return']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Positive Headline Percentage', color='g')
ax2.set_ylabel('Return', color='b')
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)
plt.show()
```
#### _Sources_

[https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html](https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html)
[https://www.kaggle.com/ngyptr/python-nltk-sentiment-analysis?scriptVersionId=904608/code](https://www.kaggle.com/ngyptr/python-nltk-sentiment-analysis?scriptVersionId=904608/code)
[https://stackoverflow.com/](https://stackoverflow.com/)
