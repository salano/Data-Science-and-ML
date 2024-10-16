import numpy as np 
from pylab import *
from scipy import stats
import matplotlib.pyplot as plt

#linear regression example

pageSpeeds = np.random.normal(3.0, 1.0, 5000)

purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 5000) ) * 3
scatter(pageSpeeds, purchaseAmount)
#run linear regress function
slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)
print(r_value ** 2)

def predict(x):
    return slope * x + intercept 

fitline = predict(pageSpeeds)
plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitline, c='r')
plt.show()

#Using polynomial regression
np.random.seed(2)# keeps a feed value so that other random operations will be deterministic
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) /pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
plt.show()

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 4))# 4 degree polynominal, p4 is the predicted or y - value
xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

#computing the r-squared
from sklearn.metrics import r2_score
r2 = r2_score(y, p4(x))
print(r2)

#multivariate regression
import pandas as pd
import statsmodels.api as sm

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')
df['Model_ord'] = pd.Categorical(df.Model).codes#the categorical function converts text values to ordinal(numeric) values we can use for analysis
X = df[['Mileage','Model_ord','Doors']]
Y = df['Price']

X1 = sm.add_constant(X)
est = sm.OLS(Y, X1).fit() #ordinary least square model

print(est.summary())

print(Y.groupby(df.Doors).mean())#get the mean price by doors




