
import numpy as np 
import pandas as pd
from pylab import *
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale 
from sklearn.cluster import KMeans
from sklearn import tree
import os, io

from IPython.display import Image 
from six import StringIO
import pydotplus

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets

#Using polynomial regression
np.random.seed(2)# keeps a feed value so that other random operations will be deterministic
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 10.0, 100) /pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
plt.show()

trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]
#plot training data
plt.scatter(trainX, trainY)
plt.show()

#plot test data
plt.scatter(testX, testY)
plt.show()

x = np.array(trainX)
y = np.array(trainY)
#generate a 8 degree polynomial
p4 = np.poly1d(np.polyfit(x, y, 8))
xp = np.linspace(0, 7, 100)
axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()

#use test dataset
testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0, 7])
axes.set_ylim([0, 200])
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
plt.show()

r2 = r2_score(testy, p4(testx))
print(r2)

r2 = r2_score(np.array(trainY), p4(trainX))
print(r2)


#implementing Naive Bayes classifier

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):

        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []

            f = io.open(path, 'r', encoding='latin1')
            for line in f:

                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True

            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message':message, 'class':classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message':[], 'class':[]})
data = data.append(dataFrameFromDirectory('D:\Python\DS\data\spam', 'Spam'))
data = data.append(dataFrameFromDirectory('D:\Python\DS\data\ham', 'Ham'))

print(data.head())

 
vectorizer = CountVectorizer()# a word tokenizer that converts words to values and count up the occurances
counts = vectorizer.fit_transform(data['message'].values) #training data (list of words and number of occurances)
classifier = MultinomialNB()
targets = data['class'].values #target data
classifier.fit(counts, targets)#create naive baye model

#testing the model
examples =['Free Money now!!!',"Hi Tanis, how about a game of chess tomorrow?"]
examples_count = vectorizer.transform(examples)#convert message to same format
predictions = classifier.predict(examples_count)
print(predictions)


#activity - do train/test dataset on pandas
# split the data into train and test set
train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)


#K-Means Clustering

#Create fake income/age cluster for N people and K clusters
def createCluster(N, K):
    np.random.seed(10)
    pointsPerCluster = np.float(N)/K
    X = []
    for i in range(K):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])

    X = np.array(X)
    return X

data = createCluster(100, 5)

model = KMeans(n_clusters=5)

#Scale the data to normalize it. Import for good results
model = model.fit(scale(data))

#look at the cluster each point was assigned to
print(model.labels_)


#visualize results
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
plt.show()

inFile = r"D:\Python\DS\data\PastHires.csv"
df = pd.read_csv(inFile, header = 0)
#map charachers and string to number
d = {'Y':1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)


#get list of features
features = list(df.columns[:6])

Y = df['Hired']
X = df[features]
clf = tree.DecisionTreeClassifier()#create model
clf = clf.fit(X, Y)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
#graph.write_pdf("EmployeeSelectionTree.pdf")

#Random Forest Classifier
clf = RandomForestClassifier(n_estimators=15)
clf = clf.fit(X, Y)

#predict employment of an employed 10-year veteran
print(clf.predict([[10, 1, 4, 0, 0, 0]]))

#predict employment of an unemployed 10-year veteran
print(clf.predict([[10, 0, 4, 0, 0, 0]]))

#SVM Clustering

#Create fake income/age cluster for N people and K clusters
def createClusterSVM(N, K):
    pointsPerCluster = np.float(N)/K
    X = []
    Y = []
    for i in range(K):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            Y.append(i)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

(X, Y) = createClusterSVM(100, 5)
plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=Y.astype(float))
plt.show()

c= 1.0
svc = svm.SVC(kernel='linear', C=c).fit(X, Y)

def plotPrediction(clf):
    xx, yy = np.meshgrid(np.arange(0, 250000, 10), np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:, 1], c=Y.astype(float))
    plt.show()

plotPrediction(svc)

#predict classification of 40 year old making 200000

print(svc.predict([[200000, 40]]))

#look up sklearn models and library