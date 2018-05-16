---
layout: post
title: Analyzing the correlation between daily Bitcoin returns & crypto using Python
---

## What is this about? 

The headline says it all. It started with a question: Is the price of bitcoin correlated to sentiment of news headlines about bitcoin and other crytocurrencies. 

Chances are you've heard about Bitcoin. Doesn't matter if your into day trading or a HODLer for life, you'll have noticed that the price of Bitcoin is volatile. In 2017, the price of Bitcoin has risen by more than 1300% while YTD it's down -36%. Price swings of 15-20% are a fairly common occurence. 

If you follow along with the media, it's clear that just like the price of bitcoin, headlines are a bit of a mixed bag. There are a lot of positive headlines talking about innovation, social impact, and positive response from goverments. There are also __A LOT__ of negative headlines about scam ICOs, bankrupt exchanges, sell-offs, and raids on illegal mining operations. 

Going back to the question: Are the two related? This post will attempt to answer that question. It will also cover how Python can be used to:

* Scrape the web for Bitcoin news headlines and store them
* Train a sentiment classifier using the Naive Bayes Algorithm to classify headlines as either negative or positive
* Ping a Bitcoin price API to calculate daily returns  
* Calculate a correlation between daily returns and the % of daily crypto headlines classified as positive
* Automate the analysis with [Launchd](http://www.launchd.info/)

## Are the two correlated?

In the absence of data, it would be logical to assume that the price of bitcoin would to some degree be affected by the sentiment of news headlines. In the presense of data, the statistical correlation between the daily price return & the % of daily crypto headlines classified as positive turns out to be __0.54__. While not a perfect linear relationship, this can be considered fairly strong. The below line chart shows this visually. The green line represents the % of daily headlines classified as positive (i.e. if 2 headlines appear on a given day & 1 is classfied as positive by the sentiment classifier, it would be a 0.5 for that day). The blue line represents daily returns calculated as: *(Today's price - Yesterday's price) / Yesterday's price*. 

![alt text](https://github.com/paolomarco/paolomarco.github.io/blob/master/images/line_chart_price_sentiment.png?raw=true "Logo Title Text 1")

## Approach

At a high level, below is the approach. 

![alt text](https://github.com/paolomarco/paolomarco.github.io/blob/master/images/approach.jpg?raw=true "Logo Title Text 1")



```python
# import libraries
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

with open('~/headlines.csv') as exisiting_file:
    existingLines = [line for line in csv.reader(exisiting_file, delimiter=',')]
    
for headline in headlines_list:
    if headline not in existingLines:
            existingLines.append(headline)
            
with open('~/headlines.csv', 'w') as f:
    pd.DataFrame(existingLines).to_csv(f,header=False,index=False)
```


