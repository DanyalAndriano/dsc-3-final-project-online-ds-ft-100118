# libraries

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from IPython.display import Image  
import pydotplus
import random
import time
import datetime as dt
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.random.seed(1)
import itertools

import scipy
from scipy import stats
from scipy.stats import boxcox
from math import sqrt

# statsmodels 
import statsmodels.api as sm
import statsmodels.api as smf
import statsmodels.stats.outliers_influence as st_inf
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import *
from statsmodels.stats.anova import anova_lm

# sklearn
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 

from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn.tree import export_graphviz, DecisionTreeClassifier 

from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc, calinski_harabaz_score, confusion_matrix
from sklearn.metrics import average_precision_score, precision_score, auc, recall_score
from sklearn.externals.six import StringIO  
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

import shap
import xgboost as xgb

# SMOTE
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

from sklearn import metrics 
from collections import Counter



