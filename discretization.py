import pandas as pd
import numpy as np
k = 8
NUM_CLASSES = 6


def get_the_centers(data):
    unique_values = np.unique(data)
    intervals = len(unique_values) / k
    t_array = unique_values[::intervals+1]
    return t_array


def boundary_points(centers):
    b = np.array([0])
    for j in range(1, k-1):
        b = np.hstack((b, (centers[j]+centers[j+1])/2))
    b = np.hstack((b, 1))
    return b


def clusters(values, points):
    length = len(values)
    cluster = np.zeros(length)

    for i in range(0, length):
        abs_array = np.absolute(points-values[i])
        cluster[i] = np.argmin(abs_array)
    return cluster


def update_clusters(cluster_array, values, centers):
    change = True
    length = len(values)
    length_centers = len(centers)

    while change:
        change = False
        for i in range(length):
            j = int(cluster_array[i])
            if j > 0 and (values[i] < centers[j]) and (values[i] - centers[j-1]) < (centers[j] - values[i]):
                cluster_array[i] = j-1
                change = True
            if j < length_centers-1 and (values[i] > centers[j]) and (centers[j+1] - values[i]) < (values[i] - centers[j]):
                cluster_array[i] = j+1
                change = True
    return cluster_array


def algorithm(values):
    centers = get_the_centers(values)
    points = boundary_points(centers)
    cluster_array = clusters(values, points)
    update_cluster = update_clusters(cluster_array, values, centers)
    final_tk = boundary_points(centers)

    return update_cluster, final_tk


def build_a_matrix(update_cluster):
    A = []
    cluster_len = len(update_cluster)
    for i in range(cluster_len):
        temp = np.zeros(k)
        temp[int(update_cluster[i])] = 1
        if len(A) == 0:
            A = temp
        else:
            A = np.vstack((A, temp))
    return A


def get_the_values(data):
    A = []
    for i in range(294):
        numpy_array = np.array(data[i].get_values())
        update_cluster, final_tk = algorithm(numpy_array)
        A_attribute = build_a_matrix(update_cluster)
        if len(A) == 0:
            A = A_attribute
        else:
            A = np.hstack((A, A_attribute))

    X = []
    for i in range(294, 300):
        numpy_array = np.array(data[i].get_values())
        if len(X) == 0:
            X = numpy_array
        else:
            X = np.vstack((X, numpy_array))

    return A, X


def update_m_matrix(A, X):
    row_a, col_a = A.shape
    M = np.zeros((NUM_CLASSES, col_a))
    row, col = X.shape
    for i in range(row):
        indices = np.nonzero(X[i])
        for index in indices:
            sum_a = np.sum(A[index], axis=0)
            M[i] += sum_a
    return M


def normalize_m_matrix(M, X):
    row, col = M.shape
    for i in range(row):
        sum_x = np.sum(X[i])
        M[i] /= sum_x
    return M

if __name__ == '__main__':
    data = pd.read_csv('data/scene-train.arff', header=None)
    A, X = get_the_values(data)
    M = update_m_matrix(A, X)
    M = normalize_m_matrix(M, X)
    print M

