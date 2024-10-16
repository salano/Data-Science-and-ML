import pandas as pd
import numpy as np
from scipy import spatial
import operator

r_cols = ['user_id', 'movie_id','rating']
ratings = pd.read_csv(r'D:\Python\DS\data\u.data', sep='\t', names=r_cols, usecols=range(3))

# Aggregate movie mean
movieProperities = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]} )

# Normalizing movie rating
MovieNumRatings = pd.DataFrame(movieProperities['rating']['size'])
movieNormalizedNumRatings = MovieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

MovieDict = {}
with open(r'D:\Python\DS\data\u.item') as f:
    temp =''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        MovieDict[movieID] = (name, list(genres), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperities.loc[movieID].rating.get('mean'))


def computeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = a[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance


def getNeighbors(MovieID, K):
    distances = []
    for movie in MovieDict:
        if (movie != movieID):
            dist = computeDistance(MovieDict[movieID], MovieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


K = 10
avgRating = 0
neighbors =  getNeighbors(1, K)
for neighbour in neighbors:
    avgRating += MovieDict[neighbour][3]
    print(MovieDict[neighbour][0] + " "+ str(MovieDict[neighbour][3]))
    avgRating /= float(K)

print(avgRating)
