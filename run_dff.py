import pandas as pd
import numpy as np
from data import *
from dff import DeepFeedForward

# get our dataframe for training
df = pd.read_csv('dataset.csv',index_col=0,header=0).dropna()

#initialize our DFF model (params given in dff.py)
dff = DeepFeedForward('ASPFWR5','ASPFWR5',0,df,[512,256,128,64])

#train our model, interactive automatically displayed
# defauly value: batch_size = 50, epochs = 20, test_size = 50
dff.train()



