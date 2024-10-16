import matplotlib.pyplot as plt
import numpy as np
from pylab import randn
from pylab import scatter, mean, dot 
from scipy.stats import norm

x = np.arange(-3, 3, 0.001)

#adjusting the axes
axes = plt.axes() #axes object
axes.set_xlim(-5, 5)#set x range
axes.set_ylim(0, 1.0)# set y range
axes.set_xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
axes.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

#changing line types and colours
axes.grid()

plt.xlabel('Greebles')# set x label
plt.ylabel('Probability')#set y label

plt.plot(x, norm.pdf(x), 'b-')
#plt.plot(x, norm.pdf(x))
#plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:') # generate another plot on the same graph
#plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r--') # generate another plot on the same graph
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r-.') # generate another plot on the same graph
#plt.plot(x, norm.pdf(x, 1.0, 0.5)) # generate another plot on the same graph
#save graph as image
#plt.savefig('D:\\Python\\testplot.png', format='png')

plt.legend(['Sneetches', 'Gacks'], loc=4) # contains the name of each graph
plt.show()

#Generating a pie chart

values = [12, 55, 4, 32, 14]
colours = ['r', 'g', 'b', 'c', 'm']
explode = [0,0,0.2,0,0]
labels = ['India','United States', 'Russia', 'China','Europe']
plt.pie(values, colors=colours, labels=labels, explode=explode)
plt.title('Student Locations')
plt.show()

#Generating Bar Charts
values = [12, 55, 4, 32, 14]
colours = ['r', 'g', 'b', 'c', 'm']
plt.bar(range(0, 5), values, color=colours)
plt.show()

#generate a acatter plot
X = randn(500)
Y = randn(500)
plt.scatter(X, Y)
plt.show()

#Generate histogram
incomes = np.random.normal(27000, 15000, 10000)# mean 27000, std 15000, data-points 10000
plt.hist(incomes, 50)#50 buckets
plt.show()

#Generating box-and-whiskers plot
uniformSkewed = np.random.rand(100) * 100 -4
high_outliers =np.random.rand(10) * 50 + 100
low_outliers =np.random.rand(10) * -50 - 100
data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
plt.boxplot(data)
plt.show()



#Calculate correlation manually
def de_mean(x):
    xmean = mean(x)
    return [xi - xmean for xi in x]

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) /(n -1)

def correlation(x, y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x, y) /stddevx / stddevy #In real life you'd check for dev by 0


pageSpeeeds =  np.random.normal(3.0, 1.0 , 1000)
purchaseAmount =  np.random.normal(50.0, 10.0 , 1000)
plt.scatter(pageSpeeeds, purchaseAmount)
plt.show()
print(covariance(pageSpeeeds, purchaseAmount))
print(np.cov(pageSpeeeds, purchaseAmount))

purchaseAmount =  np.random.normal(50.0, 10.0 , 1000) / pageSpeeeds
plt.scatter(pageSpeeeds, purchaseAmount)
plt.show()
print(covariance(pageSpeeeds, purchaseAmount))

print(correlation(pageSpeeeds, purchaseAmount))

#correlation the numpy way

print(np.corrcoef(pageSpeeeds, purchaseAmount))

#conditional probability
np.random.seed(0)

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases =  {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}

totalPurchase = 0
for _ in range(100000):
    ageDecade = np.random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = float(ageDecade) / 100.0
    totals[ageDecade] +=1
    if np.random.random() < purchaseProbability:
        totalPurchase +=1
        purchases[ageDecade] +=1