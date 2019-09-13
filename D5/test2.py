# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:22:39 2019

@author: escc
"""

def show_img(first_link):
    from PIL import Image
    from io import BytesIO
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    response = requests.get(first_link)
    
    if response.status_code == requests.codes.ok:
            img = Image.open(BytesIO(response.content))
            # Convert img to numpy array
            plt.imshow(img)
            plt.show()

target_url = 'https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt'

import requests
response = requests.get(target_url)
data = response.text

# 用 request 傳送回來的資料不會認得斷行符號
print(len(data))
data[0:100]


# 找到換行符號，用該符號做字串分割後，把它拿掉
split_tag = '\n'

data = data.split(split_tag)
print(len(data))
data[0]

import pandas as pd

var_array_arrange_data=[]

for i in range(len(data)):
    var_array_arrange_data.append(data[i].split('\t'))
    
df = pd.DataFrame(var_array_arrange_data)

a = df.head(5)
for p in a[1]:
    show_img(p)
