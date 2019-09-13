# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:49:04 2019

@author: escc
"""

import pandas as pd
import numpy as np

country = np.linspace(1, 10, 10);

people = np.random.randint(6000,20000,10);

data = {'國家': country,
        '人口': people}


data = pd.DataFrame(data)

data[data.index==data['人口'].idxmax()]