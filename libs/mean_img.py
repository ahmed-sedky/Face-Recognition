import numpy as np
import matplotlib.pyplot as plt
def get_mean_img (images):
    crop_images = []
    height, width = images[0].shape
    for image in images:
        image = image.flatten()
        crop_images.append(image)
    mean_image= np.mean(crop_images,axis=0)
    mean_image = np.reshape(mean_image, (height, width))
    return mean_image