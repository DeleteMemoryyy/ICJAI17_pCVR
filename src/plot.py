import os
import sys
import arrow as ar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyplotz.pyplotz import PyplotZ
from palettable.colorbrewer.sequential import Blues_9, BuGn_9, Greys_3, PuRd_5
import warnings

from dataloader import Dataloader

warnings.filterwarnings('ignore')

sys.path.append(r'C:\Users\xtliu\Desktop\CTR\src')
os.chdir(r'C:\Users\xtliu\Desktop\CTR\src')

plt.style.use('fivethirtyeight')
pltz = PyplotZ()
data_loader = Dataloader()
data_loader.load_file()

data = data_loader.data
