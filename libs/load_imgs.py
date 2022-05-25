import cv2
import os
def load_images_from_folder(folder):
    training_images = []
    test_images = []
    saving_indecies = []
    for root, _ , files in os.walk(folder):
        cnt = 0 
        for file in files:
            img = cv2.imread(os.path.join(root,file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                if (cnt < 8):
                    training_images.append(img_gray)
                else:
                    test_images.append(img_gray)
                    person = os.path.basename(file)
                    prefix = person.rpartition('.')[0]
                    prefix = prefix.rsplit("_", 1)[-1]
                    saving_indecies.append(prefix)
            cnt += 1
    print(saving_indecies)
    return saving_indecies,training_images,test_images