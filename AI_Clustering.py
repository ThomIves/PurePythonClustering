import matplotlib.pyplot as plt
import random
import math
import sys
import time
import json
import os


class Clusters_AI:
    def __init__(self, pnts):
        self.pnts = pnts
        self.num_pnts = len(self.pnts)
        self.num_dmns = len(self.pnts[0])

        chck_ok = sum([1 for pnt in pnts if len(pnt) != len(pnts[0])]) == 0
        err_msg = 'NOT ALL POINTS HAVE THE SAME DIMENSIONS!'
        assert chck_ok, err_msg

        self.clrs = [
            'blue', 'red', 'yellow', 'cyan', 'magenta', 'black',
            'pink', 'purple', 'brown', 'orange', 'violet', 'turquoise',
            'gray', 'white', 'navy', 'khaki', 'gold', 'lime', 'darkorange',
            'green']

    def get_mins_and_maxs(self, indices):
        mins = [1e10] * self.num_dmns
        mids = [0] * self.num_dmns
        maxs = [-1e10] * self.num_dmns

        for index in indices:
            for j in range(self.num_dmns):
                if self.pnts[index][j] < mins[j]:
                    mins[j] = self.pnts[index][j]
                if self.pnts[index][j] > maxs[j]:
                    maxs[j] = self.pnts[index][j]
                mids[j] = (maxs[j] + mins[j]) / 2.0

        return mins, mids, maxs

    def get_distance_between_two_points(self, arr1, arr2):
        sq_dist = 0
        for i in range(len(arr1)):
            sq_dist += (arr1[i] - arr2[i])**2

        return sq_dist**0.5

    def find_closest_neighbors(self, indices):
        num_pnts = len(indices)
        prxm = {}
        grps = {}
        keys_list = list(grps.keys())
        num_grps_keys = len(keys_list)

        while num_grps_keys != num_pnts:
            minD = 1.0e30
            for i in indices:
                if i in keys_list:
                    continue
                for j in indices:
                    if i == j:
                        continue
                    dist = self.get_distance_between_two_points(
                        self.pnts[i], self.pnts[j])
                    if dist < minD:
                        closest_i = i
                        closest_j = j
                        minD = dist

            grps[closest_i] = [closest_j, minD]
            prxm[minD] = [closest_i, closest_j]
            keys_list = list(grps.keys())
            num_grps_keys = len(keys_list)

        indices_grps = list(prxm.values())

        return indices_grps, prxm

    def regroup_indices(self):
        indices_del = []
        for i in range(len(self.indices_grps) - 1):
            for j in range(i + 1, len(self.indices_grps)):
                for element in self.indices_grps[j]:
                    if element in self.indices_grps[i]:
                        self.indices_grps[i] += self.indices_grps[j]
                        self.indices_grps[i] = list(set(self.indices_grps[i]))
                        indices_del.append(j)

        indices_del = list(set(indices_del))
        for index in sorted(indices_del, reverse=True):
            del self.indices_grps[index]

    def find_radius_for_each_group(self):
        self.radii_D = {}
        radius = 0.0

        for indices_grp in self.indices_grps:
            _, prxm = self.find_closest_neighbors(indices_grp)
            radius = max(list(prxm.keys()))
            self.radii_D[radius] = indices_grp

        self.radii_list = list(self.radii_D.keys())
        self.radii_list.sort(reverse=True)

    def grow_groups_within_radii(self):
        for radius in self.radii_list:
            for grp_index in self.radii_D[radius]:
                for pnt_index in range(self.num_pnts):
                    if grp_index == pnt_index:
                        continue
                    distance = self.get_distance_between_two_points(
                        self.pnts[grp_index],
                        self.pnts[pnt_index])
                    if distance <= 1.2 * radius:
                        if pnt_index not in self.radii_D[radius]:
                            self.radii_D[radius].append(pnt_index)

    def report_groupings(self):
        print(f'Latest groupings following regrouping:')
        for radius in self.radii_list:
            indices = self.radii_D[radius]  # ["indices"]
            # mids = self.radii_D[radius]["mids"]
            print(f'\tRadius: {radius}, indices: {indices}')  # , mids: {mids}')

        print()

    def determine_clusters(self):
        indices = list(range(len(self.pnts)))
        self.indices_grps, prxm = self.find_closest_neighbors(indices)
        print('Initial pairings:')
        for prox in prxm:
            print(f'\tDistance is {prox} between {prxm[prox]}')
        print()
        self.regroup_indices()
        self.find_radius_for_each_group()
        self.report_groupings()

        radii_list_delta = True
        len_radii_list = len(self.radii_list)

        while radii_list_delta:
            self.grow_groups_within_radii()
            self.regroup_indices()
            self.find_radius_for_each_group()
            new_length_radii_list = len(self.radii_list)
            radii_list_delta = new_length_radii_list - len_radii_list
            len_radii_list = new_length_radii_list
            self.report_groupings()

    def plot_clusters(self):
        X = []
        Y = []
        for grp in self.indices_grps:
            X.append([])
            Y.append([])
            for index in grp:
                X[-1].append(self.pnts[index][0])
                Y[-1].append(self.pnts[index][1])

        for i in range(len(X)):
            plt.scatter(X[i], Y[i], c=self.clrs[i])

        plt.xlabel('X Vals')
        plt.ylabel('Y Vals')
        plt.title('Fake Data Clusters')
        plt.show()


###############################################################################

class CreateFakeData:
    def __init__(self, seeds, half_range=2, points_per_cluster=10):
        self.seeds = seeds
        self.half_range = half_range
        self.points_per_cluster = points_per_cluster

    def __get_random_point_from_point__(self, arr, half_range):
        number_of_dimensions = len(arr)

        pt = []
        for i in range(number_of_dimensions):
            var = random.uniform(-half_range, half_range)
            pt.append(arr[i]+var)

        return pt

    def __get_random_point_in_range__(self, mins, maxs):
        pt = []
        for i in range(len(mins)):
            pt.append(random.uniform(mins[i], maxs[i]))

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
def scatter_plot_points(pts):
    X = []
    Y = []

    for i in range(len(pts)):
        X.append(pts[i][0])
        Y.append(pts[i][1])

    plt.scatter(X, Y)
    plt.xlabel('X Vals')
    plt.ylabel('Y Vals')
    plt.title('Fake Data Points')
    plt.show()


def line_plot(X, Y, x_label, y_label, title):
    plt.plot(X, Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


###############################################################################
# Setup Data
clr_arr = ['blue', 'red', 'yellow', 'green', 'cyan', 'magenta']
seeds = [[3, 8], [8, 3], [3, 3]]  # , [10, 10], [17, 10]]  # , [17, 3]]
hr = 2
ppc = 40
max_stretch = False

# Create Fake Data
file_name = '10-1_pnts.dat'  # '40_3-8_pnts.dat'
if os.path.exists(file_name):
    with open(file_name, 'r') as f:
        pnts = json.load(f)
else:
    fake_data = CreateFakeData(seeds, half_range=hr, points_per_cluster=ppc)
    pnts = fake_data.create_fake_data()
    # scatter_plot_points(pnts)
    with open(file_name, 'w') as f:
        json.dump(pnts, f)

cAI = Clusters_AI(pnts)
cAI.determine_clusters()
cAI.plot_clusters()


###############################################################################
# Old Code Yard
# if self.info:
#             keys_list = list(grps.keys())
#             keys_list.sort()
#             print(f'\nFinished building {len(keys_list)} groups.')
#             for index in keys_list:
#                 print(f'Group {index} closest to {grps[index]}')
#             print()

#             dist_list = list(prxm.keys())
#             dist_list.sort()
#             print(f'\nGroups sorted by {len(dist_list)} Proximities.')
#             for dist in dist_list:
#                 print(f'Distance is {dist} between {prxm[dist]}.')

# if self.info:
#             print()
#             for values in self.indices_grps:
#                 print(f'{values}')

# print('\n', indices_del) if self.info else None
