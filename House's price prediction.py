import  warnings
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import  LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from colorama import Fore, Style
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
scriptFolder = os.path.dirname((os.path.abspath(__file__)))
csvPath = scriptFolder + "/House Prices.csv"
regressorFileName = "Regressor 1 .sav"
regressorFilePath = scriptFolder + "/" + regressorFileName

df = pd.read_csv(csvPath, delim_whitespace=True, header=None)
dataset = df.values

print("\n Data set Shape:", df.shape[0], "Records &", df.shape[1], "Columns")

X = dataset[:, 0:13]
y = dataset[:, 13]

column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df.columns = column_names

correlation_matrix = df.corr()

