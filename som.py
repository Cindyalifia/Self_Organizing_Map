
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_excel('Dataset.xls', header=None)

# Split comma 
def splitComma(array) :
    listt = []
    objek = []
    for i in range (len(array)) :
        item = ''.join(array.iloc[i].values)
        name, n = item.split(",")
        listt.append(float(name))
        listt.append(float(n))
        objek.append(listt)
        listt = []
    return objek

# Inisiasi weights
def inisiasi(n) :
  list = []
  for i in range (n):
    list.append(np.random.uniform(3, 17, 2))
  return list

def euclidean(df, weights, i) :
  clust = []
  for j in range (len(weights)) :
    clust.append(math.pow((df.iloc[i,0] - weights[j][0]), 2) + math.pow((df.iloc[i,1] - weights[j][1]), 2))
  return clust

# Define all weights of neighborhood J
def neighbour(winnerJ, neighbour, weights) :
  neighborhood = math.ceil(neighbour)
  w = []
  # index bawah
  bawah = 0
  if winnerJ >= neighborhood :
    bawah = winnerJ - neighborhood 
    
  # index atas
  atas = winnerJ + neighborhood
  if atas >= (len(weights)-1) :
    atas = len(weights)-1
   
  # weights
  if atas < len(weights) :
    w = weights[bawah:atas+1]
  else :
    w = weights[bawah:]
  return (w, bawah, atas)

def updateBobot(neighborhood, weights, w, winnerJ, lr, indeksObjek) :
  # Update all of neighborhood J
  temp = []
  for i in range (len(w)) :
    Tji = math.exp( 
        -( math.pow(
           math.pow( (weights[winnerJ][0] - w[i][0]), 2) + 
           math.pow( (weights[winnerJ][1] - w[i][1]), 2) , 2) / (2*math.pow(neighborhood,2)) ) )
    matriks = [df.iloc[indeksObjek, 0] - w[i][0] , df.iloc[indeksObjek, 1] - w[i][1]]
    deltaWji = [matriks[0]*lr*Tji , matriks[1]*lr*Tji]
    updateW = [ (w[i][0] + deltaWji[0]) , (w[i][1] + deltaWji[1]) ]
    temp.append(updateW)
  return temp

# Change all weights
def changes(weights, y, bawah, atas) :
  j = 0
  asd = weights[:]
  for i in range (bawah, atas+1) :
    asd[i] = y[j]
    j += 1
  return asd

"""# Preprocessing"""

dataset = splitComma(dataset)
df = pd.DataFrame(dataset)
df

plt.scatter(df.iloc[:,0], df.iloc[:,1], color = 'yellow')

"""**Inisiasi**"""

weights = inisiasi(15)
neighborhood = 2
lr = 0.004
maxIterate = 1000

w = pd.DataFrame(weights)
plt.scatter(w.iloc[:,0], w.iloc[:,1], color = 'red')

"""# **SOM Algorihtm**

**Loop**
"""

i = 0
while i <= maxIterate and lr >= 0.000001 and neighborhood >= 0.25: # i = index df (dataset) 
  #for j in range (len(df)) :
  cluster = []
  for j in range (len(df)) :
    euclid = euclidean(df, weights, j) # j is an index from object
    winnerJ = np.argmin(euclid) # winnerJ is index of neuron's winner  
    cluster.append(winnerJ)
    weightNeighbour, bawah, atas = neighbour(winnerJ, neighborhood, weights) # Get all of neighborhood J
    y = (updateBobot(neighborhood, weights, weightNeighbour, winnerJ, lr, j)) # Get update weights, but didn't update yet
    weights = changes(weights, y, bawah, atas)
  i += 1
  lr = lr * (math.exp(-i/maxIterate))
  neighborhood = neighborhood*(math.exp(-i/maxIterate) )

i

w = pd.DataFrame(weights)
plt.scatter(df.iloc[:,0], df.iloc[:,1], color = 'yellow')
plt.scatter(w.iloc[:,0], w.iloc[:,1], color = 'red')

clusterResult = pd.DataFrame(cluster)      
clusterResult.to_csv("Result.csv", sep=',')



