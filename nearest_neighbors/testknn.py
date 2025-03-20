import numpy as np
import nearest_neighbors
import time

batch_size = 16
num_points = 81920
K = 16
pc = np.random.rand( num_points, 3).astype(np.float32)

# nearest neighbours
start = time.time()
neigh_idx = nearest_neighbors.knn(pc, pc, K, omp=True).astype(np.int32)
print(time.time() - start)


