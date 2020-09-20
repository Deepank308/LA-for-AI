import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

CONVERGENCE_DELTA = 0.0001

def load_dataset(class_limit=100, class_size=10):
    """
    loads the data set and returns the data & class labels
    """
    (data, classes), (_, _) = mnist.load_data()

    class_count = []
    for i in range(10):
        class_count.append(0)

    # get only 100 images of each class
    cut_data = []
    for d, c in zip(data, classes):
        if class_count[c] >= class_limit:
            continue
        cut_data.append(d)
        class_count[c] += 1
    
    cut_data = np.array(cut_data)
    cut_data = np.reshape(cut_data, (cut_data.shape[0], 784))
    cut_data = cut_data.astype(float) / 255

    cut_classes = np.arange(class_size)

    return cut_data, cut_classes, classes


def load_test_dataset(data_size=50):
    """
    loads the test data and normalizes the values
    """
    (_, _), (tx, ty) = mnist.load_data()

    choice = np.random.randint(tx.shape[0], size=data_size)
    test_x = tx[choice, :]
    test_x = np.reshape(test_x, (test_x.shape[0], 784))
    test_x = test_x.astype(float) / 255

    test_y = ty[choice]

    return test_x, test_y


def distance(point, center):
    """
    returns euclidean distance between two data points
    """
    
    return np.linalg.norm(point - center)


def update_centroid(centers, clusters, data):
    """
    returns the mean center of clusters
    """
    prev_centers = np.copy(centers)
    if_convergance = True
    for i, c_d in enumerate(clusters):
        if c_d.shape[0] == 0:
            centers[i] = np.zeros(data.shape[1])
            continue
        centers[i] = np.average(data[c_d, :], axis=0)
        
        if np.linalg.norm(prev_centers[i] - centers[i]) > CONVERGENCE_DELTA:
            if_convergance = False

    return centers, if_convergance


def train_kmeans(data, classes, centroids, NUM_ITR=10):
    """
    trains the kmeans for NUM_ITR iterations and returns cluster & corres. mean
    """
    # print('Training KMeans...')
    iteration = 0
    while True:
        # print('Iteration #: {}'.format(iteration + 1))
        iteration += 1

        clusters = [np.array([], dtype=np.int64) for i in range(classes.shape[0])]
        for idx, d in enumerate(data):
            dist = np.array([], dtype=np.float)

            # find the nearest cluster mean
            for c in centroids:
                dist = np.append(dist, distance(d, c))
            num = np.argmin(dist)
            
            clusters[num] = np.append(clusters[num], idx)

        # update mean of clusters
        centroids, converged = update_centroid(centroids, clusters, data)
        if converged:
            break
    
    return clusters, centroids


def test_kmeans(test_x, test_y, centers, label, clusters):
    """
    cluster prediction based on the majority 
    """
    class_cluster = []
    for c in clusters:
        cluster_data = label[c]
        cluster_label = np.bincount(cluster_data).argmax()
        class_cluster.append(cluster_label)

    predicted_y = np.array([])
    for idx, td in enumerate(test_x):
        dist = np.array([], dtype=np.float)

        # find the nearest cluster mean
        for c in centers:
            dist = np.append(dist, distance(td, c))
        num = np.argmin(dist)
        predicted_y = np.append(predicted_y, class_cluster[num])

    positive = np.where(predicted_y == test_y)[0].shape[0]
    accuracy = positive/test_y.shape[0]

    return accuracy


def calc_j_cluster(centers, clusters, data):
    j_clust = 0.0

    for center_num, idx in enumerate(clusters):
        clust_data = data[idx, :]
        for d in clust_data:
            j_clust += np.linalg.norm(centers[center_num] - d)**2
    
    j_clust /= data.shape[0]

    return j_clust


def main():
    sample = int(input('Sample choice? Random-0/From_data-1? '))
    for k in range(5, 21):
        data, classes, label = load_dataset(class_limit=100, class_size=k)

        # sample random cluster means
        if sample == 0:
            centroids = np.random.rand(classes.shape[0], data.shape[1])
        else:
            centroids = data[np.random.randint(data.shape[0], size=classes.shape[0]), :]
        
        # train the KMeans
        clusters, centroids = train_kmeans(data, classes, centroids)
        
        # part (c)
        j_clust = calc_j_cluster(centroids, clusters, data)
        print('k: {}    J-clust: {}'.format(k, j_clust))
        
    # part (a)
    _, axarr = plt.subplots(4, 5)
    centers = centroids*255

    sz = centers.shape[0]
    for i, c in enumerate(centers):
        c = np.reshape(c, (28, 28))
        axarr[i//5, i%5].imshow(c, cmap='gray')
        axarr[i//5, i%5].axis('off')

    # plt.savefig('thumbnail-{}.jpg'.format(sample))

    # part (b)
    test_x, test_y = load_test_dataset(data_size=50)
    test_acc = test_kmeans(test_x, test_y, centroids, label, clusters)
    print('Test accuracy: {}%'.format(test_acc*100)) 


if __name__ == '__main__':
    main()
