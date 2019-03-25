import pandas as pd
import numpy as np
from data import *


mod = DataGenerator()

df = mod.sub_datasets['MVOLE added']

x,y = mod.train_split_multiInput(df,'ASPFWR5',60)


