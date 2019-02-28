# K-Means Clustering

# Importing the libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys, random

clr_arr = ['blue','red','yellow','green','cyan']

###############################################################################

class CreateFakeData(object):
    def __init__(self, seeds, half_range=2, points_per_cluster=10):
        self.seeds = seeds
        self.half_range = half_range
        self.points_per_cluster = points_per_cluster

    def __get_random_point_from_point__(self, arr, half_range):
        number_of_dimensions = len(arr)

        pt = []
        for i in range(number_of_dimensions):
            var = random.uniform(-half_range,half_range)
            pt.append(arr[i]+var)

        return pt

    def __get_random_point_in_range__(self, mins, maxs):
        pt = []
        for i in range(len(mins)):
            pt.append(random.uniform(mins[i],maxs[i]))

        return pt

    def create_fake_data(self):
        # Initialize fake data and centroids
        KNN_A = []

        for seed in self.seeds:
            for i in range(self.points_per_cluster): 
                KNN_A.append(self.__get_random_point_from_point__(
                    seed, self.half_range))

        return KNN_A

###############################################################################

# Setup Fake Data
clr_arr = ['blue','red','yellow','green','cyan','magenta']
seeds = [[3,10], [10,3], [3,3], [10,10], [17,6]]
half_range = 2

# Create Fake Data
fake_data = CreateFakeData(seeds)
KNN_A = fake_data.create_fake_data()

###############################################################################

# Fitting K-Means to the dataset 
kmeans = KMeans(n_clusters = len(seeds), init = 'random', random_state = 42)
y_kmeans = kmeans.fit_predict(KNN_A)
print(kmeans.inertia_)

groups = y_kmeans.tolist()
KNN_C = kmeans.cluster_centers_.tolist()

# Plot the results

grps = {}
for i in range(len(seeds)):
    grps[i] = {'X':[], 'Y':[]}

Xc = []; Yc = []

for arr in KNN_C:
    Xc.append(arr[0]) # print(Xc)
    Yc.append(arr[1]) # print(Yc)

for i in range(len(groups)):
    grps[groups[i]]['X'].append(KNN_A[i][0])
    grps[groups[i]]['Y'].append(KNN_A[i][1])

#######
for i in range(len(grps)):
    plt.scatter(grps[i]['X'], grps[i]['Y'], c=clr_arr[i])
plt.scatter(Xc, Yc, c='black')
plt.xlabel('X Vals')
plt.ylabel('Y Vals')
plt.title('The Title')
plt.show()