import cv2
import os
def load_images_from_folder(folder):
    training_images = []
    test_images = []
    for root, dirs, files in os.walk(folder):
        cnt = 0 
        cnt_train = 0
        for file in files:
            img = cv2.imread(os.path.join(root,file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                if (cnt < 8):
                    training_images.append(img_gray)
                    cnt_train += 1
                else:
                    test_images.append(img_gray)
            cnt += 1
    return training_images,test_images