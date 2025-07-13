import  warnings
import os
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
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

plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20,20))
fig.suptitle("Histogram of Features", fontsize = 20)

for i in [13,14,15]:
    fig.delaxes(axes.flatten()[i])

for i,col in enumerate(df.columns):
    ax = axes.flatten()[i]
    df[col].hist(ax = ax, bins=30, color = 'skyblue', edgecolor = 'black')
    ax.set_title(col)
    ax.grid(False)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

XTraining, XTest, yTraining, yTest = train_test_split(X,y, test_size=0.2, random_state=42)
print(Fore.BLUE, "\n Evaluating different Regressors using Train-Test Split Method. Please wait...", Fore.BLACK)

regressors = []
regressors.append(("RF", "Random Forest", RandomForestRegressor(n_estimators=1000, random_state=42)))
regressors.append(("XGB", "Extreme Gradient Boosting", XGBRegressor()))
regressors.append(("CART", "Decision Tree", DecisionTreeRegressor()))
regressors.append(("SVM", "Support Vectore Machine", SVR()))
regressors.append(("KNN", "K-Nearest Neighbour", KNeighborsRegressor()))
regressors.append(("MLR", "Multiple Linear Regression", LinearRegression()))
regressors.append(("MLP", "Multi-layer Perceptron",
                  MLPRegressor(hidden_layer_sizes=(13,6), activation="relu", solver="adam",
                               max_iter=500, batch_size=10))
                  )
results = []
for code, name, regressor in regressors:
    regressor.fit(XTraining, yTraining)
    yPredicted = regressor.predict(XTest)
    r2 = r2_score(yTest, yPredicted)
    mae = mean_absolute_error(yTest, yPredicted)
    mse = mean_squared_error(yTest, yPredicted)
    rmse = np.sqrt(mse)

    results.append((code, name, regressor, round(r2,2), round(mae, 2), round(mse, 2), round(rmse, 2)))

results.sort(key=lambda i : i[3])
print(Fore.RED,"\n# Regressor \t\t\t\t\t\tTest RMSE (K$) \t Test MAE(k$) \t Test R-Squared", Fore.WHITE)
print("-------------------------------------------------------------------------------------------------")
i =1
for r in results:
    s = r[0] + " - " +r[1]
    print("", i, " ", s, " "*(36-len(s)), " ", r[3], "\t\t\t", r[4], "\t\t\t", r[5])
    i+=1
bestRegressor = results[0][2]
bestRegressorCode = results[0][0]
bestRegressorName = results[0][1]
print(Fore.MAGENTA, "\n Best Regressor: ", bestRegressorCode, "-", bestRegressorName)
print(Style.RESET_ALL)

bestRegressor.fit(X,y)

pickle.dump(bestRegressor, open(regressorFilePath, "wb"))
print(Fore.BLUE + "\n Final Regressor saved to Disk As: '" + regressorFileName + "'")
