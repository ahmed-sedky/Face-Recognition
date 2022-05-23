import matplotlib
import numpy as np
import matplotlib.pyplot as plt
def get_mean_img (images):
    crop_images = []
    for i, image in enumerate(images):
    # crop_img , no_faces = face_detect.detect_face(image)
        if i==0:
            height, width = image.shape
        image = image.flatten()
        crop_images.append(image)
    mean_image= np.mean(crop_images,axis=0)
    mean_image = np.reshape(mean_image, (height, width))
    # plt.imshow(mean_image,cmap="gray")
    # plt.show()
    return mean_image