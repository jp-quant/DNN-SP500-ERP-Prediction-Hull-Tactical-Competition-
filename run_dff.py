import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import *
from dff import DeepFeedForward
from sklearn.metrics import mean_squared_error as calculate_mse










# just an mse report function that prints out the prediction report
def model_mse_report(dff_model):
    prediction = dff_model.predict()

    unscaled_mse = calculate_mse(prediction['unscaled'].dropna()['test'],prediction['unscaled'].dropna()['prediction'])
    scaled_mse = calculate_mse(prediction['scaled'].dropna()['test'],prediction['scaled'].dropna()['prediction'])
    model_name = dff_model.model_name

    print('---< '+model_name+'\'s Prediction MSE >---')
    print('Unscaled:',str(unscaled_mse),'  Scaled:',str(scaled_mse))

#plot our prediction, input = df.tail(test_size) (test_size = 50 on default)
def plot_prediction(dff_model):
    prediction = dff_model.predict()

    plt.plot(prediction['unscaled'])
    plt.show()
    

mod  = DataManager()
#x,y = mod.prepare_XY_multiInput(df,'ASPFWR5',50)

# get our dataframe for training
df = mod.sub_datasets['MVOLE added']

# WE WILL SHIFT ASPFWR5 DOWNWARD BY 5, ACTING AS A FEATURE FOR POTENTIAL CORRELATION SIGNAL
# MEANING THAT THE NEW ASPFWR0 COLUMN = S&P500 RETURN ON THAT EXACT DAY
df['ASPFWR0'] = df['ASPFWR5'].shift(5)
#df = df.ffill() #temporary estimator since little N/A

#initialize our DFF model (params given in dff.py)
dff = DeepFeedForward('ASPFWR5','ASPFWR5',0,df,[512,256,128,64])

#train our model, interactive automatically displayed
# defauly value: batch_size = 32, epochs = 10, test_size = 32
dff.train(save_model=True)
model_mse_report(dff)
plot_prediction(dff)



