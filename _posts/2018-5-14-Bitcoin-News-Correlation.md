---
layout: post
title: Looking at the relationship between Bitcoin returns & Crypto news sentiment
---

## What is this about?

Unless you've been living under a rock, you've heard about Bitcoin and maybe even own some.  

## Is there a relationship?

## Approach

Here is the script to get the headlines

```python
# import libraries
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime as dt
import requests
import csv

# specify the url
quote_page = "https://www.coindesk.com/"
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


