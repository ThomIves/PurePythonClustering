import matplotlib.pyplot as plt
import random, sys, math
clr_arr = ['blue','red','yellow','green','cyan']

def get_random_point_from_point(arr, range):
    varX = random.uniform(-range[0],range[0])
    varY = random.uniform(-range[1],range[1])
    pt = [arr[0]+varX, arr[1]+varY]

    return pt

def get_mins_and_maxs(KNN_A):
    # Find min and max to get initial random centroids range
    minX = 1e10; maxX = -1e10; minY = 1e10; maxY = -1e10
    for arr in KNN_A:  # print(minX, maxX, minY, maxY)
        if arr[0] < minX: minX = arr[0]
        if arr[0] > maxX: maxX = arr[0]
        if arr[1] < minY: minY = arr[1]
        if arr[1] > maxY: maxY = arr[1]

    return minX, maxX, minY, maxY

def get_avg_of_pts(KNN_A):
    Xtot = 0; Ytot = 0; num = len(KNN_A)
    for arr in KNN_A:  # print(minX, maxX, minY, maxY)
        Xtot += arr[0]
        Ytot += arr[1]

    return [Xtot / num, Ytot / num]

def get_distribution_parameters_of_pts(KNN_A):
    dims = len(KNN_A[0]) 
    means = [0] * dims
    stds  = [0] * dims
    num_pts = len(KNN_A)

    for arr in KNN_A:
        for i in range(dims):
            means[i] += arr[i]

    for i in range(dims): means[i] /= num_pts

    for arr in KNN_A:
        for i in range(dims):
            stds[i] += (means[i] - arr[i])**2

    for i in range(dims): stds[i] = (stds[i]/num_pts)**0.5

    return [means, stds]
    
def get_random_point_in_range(minX, maxX, minY, maxY):
    varX = random.uniform(minX,maxX)
    varY = random.uniform(minY,maxY)
    pt = [varX, varY]

    return pt

def get_distance_between_two_points(arr1, arr2):
    sq_dist = (arr1[0] - arr2[0])**2 + \
              (arr1[1] - arr2[1])**2

    return sq_dist**0.5

def group_points_by_centroid(grps, KNN_C, KNN_A):
    for pta in KNN_A:
        minD = 1e10
        for i in range(len(KNN_C)):
            ptc = KNN_C[i]
            dist = get_distance_between_two_points(ptc, pta)
            if dist < minD:
                closest_pt = i
                minD = dist

        grps[closest_pt]['As'].append(pta)
        grps[closest_pt]['X'].append(pta[0])
        grps[closest_pt]['Y'].append(pta[1])

    return grps

def determine_inertia(grps, KNN_C, KNN_A):
    inertia = 0
    for i in range(len(grps)):
        for pt in range(len(grps[i]['As'])):
            dist = get_distance_between_two_points(
                    KNN_C[i], grps[i]['As'][pt])
            inertia += (dist) ** 2

    return inertia

def update_centroids(grps):
    KNN_C_New = []
    for i in range(len(grps)):
        x_sum = 0; y_sum = 0
        num = len(grps[i]['As'])
        if num == 0: # if no members are in the centroid group ...
            out = get_mins_and_maxs(KNN_A)
            KNN_C_New.append( # assign that centroid a new random location
                get_random_point_in_range(out[0],out[1],out[2],out[3]))
            continue # then continue
        for j in range(num):
            x_sum += grps[i]['X'][j]
            y_sum += grps[i]['Y'][j]
        KNN_C_New.append([x_sum/num, y_sum/num])

    return KNN_C_New

def find_As_delta(KNN_C, KNN_C_New):
    dist_sum = 0
    for i in range(len(KNN_C)):
        dist_sum += get_distance_between_two_points(KNN_C[i], KNN_C_New[i])

    return dist_sum

def initial_dispersement_of_centroids(KNN_A, num_cts, method='mean_2sig_rng'):
    #  method='mean_std_spiral'
    dist_params = get_distribution_parameters_of_pts(KNN_A)
    means = dist_params[0]
    stds  = dist_params[1]
    two_sig  = []
    for element in stds:
        two_sig.append(element*2.0)

    num_dims = len(means)
    KNN_C = []

    if method == 'mean_2sig_rng':
        for i in range(num_cts):
            KNN_C.append(
                get_random_point_from_point(
                    means.copy(),two_sig.copy()))

        return KNN_C

    elif method == 'mean_std_spiral':
        KNN_C.append(means.copy())
        cnt = 1; radius = 1
        while True:
            for i in range(num_dims):
                KNN_C.append(means.copy())
                KNN_C[-1][i] += radius * stds[i]
                cnt += 1
                if cnt >= num_cts:
                    return KNN_C

            for i in range(num_dims):
                KNN_C.append(means.copy())
                KNN_C[-1][i] -= radius * stds[i]
                cnt += 1
                if cnt >= num_cts:
                    return KNN_C

            radius += 1
        

# Initialize fake data and centroids
KNN_A = []
seeds = [[3,10], [10,3], [3,3], [10,10], [17,6]]

for seed in seeds:
    for i in range(10): 
        KNN_A.append(get_random_point_from_point(seed, [2,2]))

###############################################################################
ans = {}
for attempt in range(10):
    KNN_C = initial_dispersement_of_centroids(KNN_A, len(seeds))

    # Loop beginning
    cnt = 0
    while cnt < 100:
        grps = {}
        for i in range(len(seeds)):
            grps[i] = {'As':[], 'X':[], 'Y':[]}

        # Find groups by closest to centroid
        grps = group_points_by_centroid(grps, KNN_C, KNN_A)

        KNN_C_New = update_centroids(grps)
        
        delta_As = find_As_delta(KNN_C, KNN_C_New)
    
        if delta_As == 0: break

        KNN_C = KNN_C_New

        cnt += 1

    #######
    ans[determine_inertia(grps, KNN_C, KNN_A)] = (KNN_C, grps)

ans_keys = sorted(ans.keys())
print('Inertia is {}'.format(ans_keys[0]))
KNN_C = ans[ans_keys[0]][0]
grps  = ans[ans_keys[0]][1]

Xc = []; Yc = []

for arr in KNN_C:
    Xc.append(arr[0])
    Yc.append(arr[1])

for i in range(len(grps)):
    plt.scatter(grps[i]['X'], grps[i]['Y'], c=clr_arr[i])
plt.scatter(Xc, Yc, c='black')
plt.xlabel('X Vals')
plt.ylabel('Y Vals')
plt.title('The Title')
plt.show()