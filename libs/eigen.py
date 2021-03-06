import numpy as np
import matplotlib.pyplot as plt

def get_eigen(cov_mat,substracted_images):
    eigen_values,eigen_vectors = np.linalg.eig(cov_mat)
    eigen_vectors = np.dot(eigen_vectors, substracted_images)
    tot_eigen_values = np.sum(eigen_values)
    accepted_variance = 0
    cnt = 0
    for eigen_val in eigen_values:
        if ( accepted_variance/ tot_eigen_values < 0.9): 
            accepted_variance += eigen_val
            cnt += 1
    eigen_vectors = eigen_vectors[:cnt , : ] 
    print ("count of eigen vectors= " , cnt)
    print ("accepted_variance= " , accepted_variance/ tot_eigen_values)
    # for eigen_vec in eigen_vectors:
    #     plt.imshow(eigen_vec.reshape(80,70),cmap="gray")
    #     plt.show()
    return eigen_vectors