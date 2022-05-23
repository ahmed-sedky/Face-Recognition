import numpy as np
def get_projection(eigen_vectors,images):
    images = np.asarray(images)
    images = np.transpose(images)
    projected_images = np.dot(eigen_vectors, images)
    projected_images = np.transpose(projected_images)
    print (projected_images.shape)
    return projected_images