import numpy as np
def get_sub_images(mean_image ,images):
    substracted_images = []
    for image in images:
        sub_img = image - mean_image
        sub_img =  np.asarray(sub_img).flatten() 
        substracted_images.append(sub_img)
    # plt.imshow(sub_image,cmap="gray")
    # plt.show()
    return substracted_images
