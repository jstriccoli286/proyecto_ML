import sys

import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier # modelo con árboles de decisión similar a XGBoost
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

import pickle

MODEL_PATH = 'boost.pkl'

#Load Scikit-learn model
loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
print(loaded_model)

##

if len(sys.argv) > 0:

    #process the arguments
    arguments = sys.argv[1:]
    for arg in arguments:
        print(arg)

    new_values = pd.DataFrame([arguments],columns=['EK','EK_DMSNR_Curve'])
    prediction = pd.DataFrame(loaded_model.predict(new_values),columns=['prediction'])
    print(prediction)

    result_df = pd.concat([new_values, prediction], axis='columns')
    print(result_df)

else:
    print('0 arguments have been received')