import matplotlib.pyplot as plt
import random
import math
import sys
import time


class KMeans:
    """
    Finds clusters of points and their centroids for a number of clusters.
    """
    def __init__(self, n_clusters=8, init='mn_2sg_rng', n_init=10, max_iter=300):
        """
        Initialization method for instantiating this class.
        ...
        Keyword Arguments:
            n_clusters {number} -- divide the data into this number of
                clusters (default: {8})
            init {str} -- the algorithm used to determine initial centroid
                locations (default: {'mn_2sg_rng'})
            n_init {number} -- the number of times to group the data into
                n_clusters - to ensure bad initial centroid locations
                don't prevent finding good clustering (default: {10})
            max_iter {number} -- the limit on the number of cycles to converge
                on centroid locations (default: {300})
            clr_arr {array} -- a convenience array for colors for plotting.
        """

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter

        self.clr_arr = ['blue', 'red', 'yellow', 'green', 'cyan', 'magenta']

    def __get_mins_and_maxs__(self, KNN_A):
        """
        Find the minimum and maximum values for all the points in KNN_A
        for all the dimensions.

        Arguments:
            KNN_A {array of arrays} -- each array in the array has values
                for each dimension of each point.
        ...
        Returns:
            [floats], [floats] -- the first array has the minimum points
                for each dimension; the second has the maximum ones.
        """
        number_of_points = len(KNN_A)
        number_of_dimensions = len(KNN_A[0])

        mins = [1e10] * number_of_dimensions
        maxs = [-1e10] * number_of_dimensions

        for i in range(number_of_points):
            for j in range(number_of_dimensions):
                if KNN_A[i][j] < mins[j]:
                    mins[j] = KNN_A[i][j]
                if KNN_A[i][j] > maxs[j]:
                    maxs[j] = KNN_A[i][j]

        return mins, maxs

    def __get_random_point_from_point__(self, arr, half_range):
        """
        Kicks a point to a new location between -half_range and +half_range
        in all dimensions.
        ...
        Arguments:
            arr {[floats]} -- an array of dimensional elements for a point.
            half_range {array} -- an array of dimensional values that are the
                negative and positive limits of a new random dimesional value.
        ...
        Returns:
            [floats] -- all dimensional values for the new point.
        """
        number_of_dimensions = len(arr)

        pt = []
        for i in range(number_of_dimensions):
            var = random.uniform(-half_range[i], half_range[i])
            pt.append(arr[i]+var)

        return pt

    def __get_random_point_in_range__(self, mins, maxs):
        """
        Create a random multidimensional point between mins and maxs
            for each dimension.
        ...
        Arguments:
            mins {[floats]} -- minimum values for each dimension
            maxs {[floats]} -- maximum values for each dimension
        ...
        Returns:
            [floats] -- all dimensional values for the new point.
        """
        pt = []
        for i in range(len(mins)):
            pt.append(random.uniform(mins[i], maxs[i]))

        return pt

    def __get_distribution_parameters_of_pts__(self, KNN_A):
        """
        Determine the statistics for each dimension of the points in KNN_A.
        ...
        Arguments:
            KNN_A {array of arrays} -- each array in the array has all values
                for each dimension of each point.
        ...
        Returns:
            [floats], [floats] -- the arrays have the means and standard
                deviations for each dimension of the points in KNN_A.
        """
        num_pts = len(KNN_A)
        number_of_dimensions = len(KNN_A[0])

        means = [0] * number_of_dimensions
        stds = [0] * number_of_dimensions

        # Determine means for each dimension
        for arr in KNN_A:
            for i in range(number_of_dimensions):
                means[i] += arr[i]
        for i in range(number_of_dimensions):
            means[i] /= num_pts

        # Determine standard deviations for each dimension
        for arr in KNN_A:
            for i in range(number_of_dimensions):
                stds[i] += (means[i] - arr[i])**2
        for i in range(number_of_dimensions):
            stds[i] = (stds[i] / num_pts)**0.5

        return means, stds

    def __get_distance_between_two_points__(self, arr1, arr2):
        """
        The function name says it all. :-)
        ...
        Arguments:
            arr1 {array of floats} -- the value of each dimension for the
                first point.
            arr2 {array of floats} -- the value of each dimension for the
                second point.
        ...
        Returns:
            array of floats -- the euclidean distance between the two points.
        ...
        Raises:
            ArithmeticError -- makes sure that each point has the same
                dimensions.
        """
        err_str = 'Error in "__get_distance_between_two_points__":\n'
        err_str += 'Point arrays do not have the same dimensions.'

        if len(arr1) != len(arr2):
            raise ArithmeticError(error_string)

        sq_dist = 0
        for i in range(len(arr1)):
            sq_dist += (arr1[i] - arr2[i])**2

        return sq_dist**0.5

    def __group_points_by_centroids__(self, grps, KNN_C, KNN_A):
        """
        Go thru all points and determine which centroid each point is
        closest to. Group the points with their closest centroids in grps.

        Arguments:
            grps {dictionary of dictionaries} -- each dictionary represents
                the points belonging to a centroid. See initial formation in
                determine_k_clusters.
            KNN_C {array of arrays} -- each array in the array has all values
                for each dimension of each centroid.
            KNN_A {array of arrays} -- each array in the array has all values
                for each dimension of each point.

        Returns:
            [dictionary of dictionaries] -- the updated relations of points
                associated to their nearest centroids.
        """
        for pta in KNN_A:
            minD = 1e10
            for i in range(len(KNN_C)):
                ptc = KNN_C[i]
                dist = self.__get_distance_between_two_points__(
                    ptc, pta)
                if dist < minD:
                    closest_centroid = i
                    minD = dist

            grps[closest_centroid]['points'].append(pta)
            if grps[closest_centroid]['centroids'] == []:
                grps[closest_centroid]['centroids'] = \
                    KNN_C[closest_centroid]

        return grps

    def __determine_inertia__(self, grps):
        """
        Calculates the inertia, which is simply the sum of the square
            distances of each point from it's current centroid.

        Arguments:
            grps {dictionary of dictionaries} -- each dictionary represents
                the points belonging to a centroid. See initial formation in
                determine_k_clusters.

        Returns:
            float -- the calculated inertia value
        """
        inertia = 0
        for i in range(len(grps)):
            for j in range(len(grps[i]['points'])):
                dist = self.__get_distance_between_two_points__(
                        grps[i]['centroids'],
                        grps[i]['points'][j])
                inertia += (dist) ** 2

        return inertia

    def __determine_master_inertia__(self, grps, KNN_A):
        """
        Calculates the master inertia, which is simply the sum of the square
            distances of each centroid to the centroid of the centroids.

        Arguments:
            grps {dictionary of dictionaries} -- each dictionary represents
                the points belonging to a centroid. See initial formation in
                determine_k_clusters.

        Returns:
            float -- the calculated inertia value
        """

        # Part of master_inertia calculations - find centroid of centroids
        # grps[i]['inertia'] += (dist) ** 2  # Needed for master_inertia
        number_of_dimensions = len(KNN_A[0])
        total_number_of_clusters = len(grps)
        total_number_of_points = len(KNN_A)

        self.total_cnt = [0] * number_of_dimensions
        self.cnt_of_cnts = [0] * number_of_dimensions
        self.clus_DOC = self.n_clusters - 1

        # Find centroid of all points
        for i in range(number_of_dimensions):
            for j in range(total_number_of_clusters):
                number_of_points_in_cluster = len(grps[j]['points'])
                for k in range(number_of_points_in_cluster):
                    self.total_cnt[i] += grps[j]['points'][k][i]

            self.total_cnt[i] /= total_number_of_points

        # Find centroid of all centroids
        for i in range(number_of_dimensions):
            for j in range(total_number_of_clusters):
                self.cnt_of_cnts[i] += grps[j]['centroids'][i]

            self.cnt_of_cnts[i] /= total_number_of_clusters

        # Find master inertia
        self.master_inertia = 0
        for i in range(len(grps)):
            dist = self.__get_distance_between_two_points__(
                grps[i]['centroids'], self.total_cnt)
            self.master_inertia += (dist / self.clus_DOC) ** 2

        # #####################################################################
        # self.clus_mod_0 = (self.master_inertia / self.inertia_)
        #  * (self.n_clusters - 1) ** 2
        # self.clus_mod_a = self.clus_mod_0 / (self.n_clusters - 1) ** 0.5
        # self.clus_mod_1 = self.master_inertia / (self.n_clusters - 1)
        # self.clus_mod_2 = self.master_inertia / (self.n_clusters - 1) ** 2
        # self.clus_mod_3 = self.master_inertia / self.inertia_
        # self.clus_mod_4 = self.clus_mod_3 / (self.n_clusters - 1)
        # self.clus_mod_5 = self.clus_mod_3 / (self.n_clusters - 1) ** 2

        # self.master_inertia = self.master_inertia ** 0.5

        # self.master_inertia /= self.inertia_**0.5

    def __update_centroids__(self, grps, KNN_A):
        """
        Once points are in groups of being closest to a certain centroid,
            the centroids can be moved to the center of those points.

        Arguments:
            grps {dictionary of dictionaries} -- each dictionary represents
                the points belonging to a centroid. See initial formation in
                determine_k_clusters.
            KNN_A {array of arrays} -- each array in the array has all values
                for each dimension of each point.

        Returns:
            array of arrays -- the updated centroid points.
            dictionary of dictionaries - the groups with updated centroid
                locations.
        """

        KNN_C_New = []
        total_number_of_clusters = len(grps)
        number_of_dimensions = len(KNN_A[0])
        mins, maxs = self.__get_mins_and_maxs__(KNN_A)

        for i in range(total_number_of_clusters):
            number_of_points_in_cluster = len(grps[i]['points'])
            if number_of_points_in_cluster == 0:
                # assign that centroid a new random location
                KNN_C_New.append(
                    self.__get_random_point_in_range__(mins, maxs))
                grps[i]['centroids'] = KNN_C_New[-1]
                continue  # then continue

            cnt_locs = [0] * number_of_dimensions

            for j in range(number_of_dimensions):
                for k in range(number_of_points_in_cluster):
                    cnt_locs[j] += grps[i]['points'][k][j]
                cnt_locs[j] /= number_of_points_in_cluster

            grps[i]['centroids'] = cnt_locs
            KNN_C_New.append(cnt_locs)

        return KNN_C_New, grps

    def __find_Arrays_delta__(self, KNN_C, KNN_C_New):
        """
        Determine the distance between the last and current centroids.

        Arguments:
            KNN_C {array of arrays} -- last centroid locations where each
                array in the array has all values for each dimension of
                each centroid.
            KNN_C_New {array of arrays} -- latest centroid locations.

        Returns:
            float -- the sum of the distances between the current and the
                last centroids.
        """
        dist_sum = 0
        for i in range(len(KNN_C)):
            dist_sum += self.__get_distance_between_two_points__(
                KNN_C[i], KNN_C_New[i])

        return dist_sum

    def __initial_disbursement_of_centroids__(
      self, KNN_A, n_clusters, method='mn_2sg_rng'):
        """
        Set initial positions for the centroids.

        Arguments:
            KNN_A {array of arrays} -- each array in the array has all values
                for each dimension of each point.
            n_clusters {integer} -- the algorithm is attempting to form the
                points into this many clusters.

        Keyword Arguments:
            method {string} -- the method chosen for establishing the initial
                centroid locations (default: {'mn_2sg_rng'})
        """
        #  method='mean_std_spiral'
        means, stds = self.__get_distribution_parameters_of_pts__(
            KNN_A)

        two_sig = []
        for element in stds:
            two_sig.append(element*2.0)

        KNN_C = []

        if method == 'mn_2sg_rng':
            for i in range(n_clusters):
                KNN_C.append(
                    self.__get_random_point_from_point__(
                        means, two_sig))

            return KNN_C

        elif method == 'mean_std_spiral':
            num_dims = len(means)
            KNN_C.append(means.copy())
            cnt = 1
            radius = 1
            while True:
                for i in range(num_dims):
                    KNN_C.append(means.copy())
                    KNN_C[-1][i] += radius * stds[i]
                    cnt += 1
                    if cnt >= n_clusters:
                        return KNN_C

                for i in range(num_dims):
                    KNN_C.append(means.copy())
                    KNN_C[-1][i] -= radius * stds[i]
                    cnt += 1
                    if cnt >= n_clusters:
                        return KNN_C

                radius += 1

    def determine_k_clusters(self, KNN_A):
        """
        The top level routine for attempting to best group the points for
        the chosen number of clusters.

        Arguments:
            KNN_A {array of arrays} -- each array in the array has all values
                for each dimension of each point.

        Returns:
            a dictionary of dictionaries -- the best clustering of points
                along with their centroid locations. This grouping was
                found to have the lowest intertia too for the n_init times
                that the clustering was determined starting from random
                initial centroid locations.
        """
        min_inertia = 1e10
        for attempt in range(self.n_init):
            KNN_C = self.__initial_disbursement_of_centroids__(
                KNN_A, self.n_clusters)

            # Loop beginning
            cnt = 0
            while cnt < self.max_iter:
                grps = {}
                for i in range(self.n_clusters):
                    grps[i] = {'centroids': [], 'points': [], 'inertia': 0}

                # Find groups by closest to centroid
                grps = self.__group_points_by_centroids__(grps, KNN_C, KNN_A)

                KNN_C_New, grps = self.__update_centroids__(grps, KNN_A)

                delta_As = self.__find_Arrays_delta__(KNN_C, KNN_C_New)

                if delta_As == 0:
                    break

                KNN_C = KNN_C_New

                cnt += 1

            #######
            current_inertia = self.__determine_inertia__(grps)

            if current_inertia < min_inertia:
                min_inertia = current_inertia
                grps_best = grps

        self.inertia_ = min_inertia
        self.__determine_master_inertia__(grps, KNN_A)

        return grps_best

    def plot_clusters(self, grps):
        """
        Plot clusters :-)

        Arguments:
            grps {dictionary of dictionaries} -- each dictionary represents
                the points belonging to a centroid. See initial formation in
                determine_k_clusters.
        """
        Xc = [self.total_cnt[0], self.cnt_of_cnts[0]]
        Yc = [self.total_cnt[1], self.cnt_of_cnts[1]]

        for i in range(len(grps)):
            Xc.append(grps[i]['centroids'][0])
            Yc.append(grps[i]['centroids'][1])

        X = []
        Y = []
        for i in range(len(grps)):
            X.append([])
            Y.append([])
            for j in range(len(grps[i]['points'])):
                X[i].append(grps[i]['points'][j][0])
                Y[i].append(grps[i]['points'][j][1])

        for i in range(len(X)):
            plt.scatter(X[i], Y[i], c=self.clr_arr[i])

        plt.scatter(Xc, Yc, c='black')
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


###############################################################################
# Setup Data
clr_arr = ['blue', 'red', 'yellow', 'green', 'cyan', 'magenta']
seeds = [[3, 10], [10, 3], [3, 3], [10, 10], [17, 10], [17, 3]]
half_range = 2

# Create Fake Data
fake_data = CreateFakeData(seeds, points_per_cluster=30)
KNN_A = fake_data.create_fake_data()
# scatter_plot_points(KNN_A)

# Find the Clusters
for i in range(2, 11):
    km = KMeans(n_clusters=i)
    grps = km.determine_k_clusters(KNN_A)
    # print(f'{i}:')
    # print(f'\t{kmeans.inertia_}')
    # print(f'\t{kmeans.master_inertia}')
    # print(f'\t{kmeans.inertia_ * kmeans.master_inertia}')
    # print(f'\t{kmeans.clus_DOC}')
    qnt = km.inertia_ / (km.master_inertia)  # / km.clus_DOC ** 3)
    print(f'{i}: {qnt}')
    # print(f'\t{kmeans.clus_mod_0}')
    # print(f'\t{kmeans.clus_mod_a}')
    # print(f'\t{kmeans.clus_mod_1} for {i} clusters.')
    # print(f'\t{kmeans.clus_mod_2} for {i} clusters.')
    # print(f'\t{kmeans.clus_mod_3} for {i} clusters.')
    # print(f'\t{kmeans.clus_mod_4} for {i} clusters.')
    # print(f'\t{kmeans.clus_mod_5} for {i} clusters.')
    # kmeans.plot_clusters(grps)

###############################################################################
