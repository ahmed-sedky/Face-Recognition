import numpy as np
def get_cov_mat(substracted_images,images):
    substracted_images = np.asarray(substracted_images)
    substracted_images_Transpose= np.transpose(substracted_images)
    cov_mat = np.dot(substracted_images, substracted_images_Transpose)
    no_images = len(images)
    cov_mat = (1/(no_images -1 )) *np.asarray(cov_mat)
    return cov_mat