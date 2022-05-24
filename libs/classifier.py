import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
def classify(projected_training_imgs,projected_test_imgs):
    y = []
    y_true = []
    for i in range (41):
        for num in range (8):
            y.append(i)
    for i in range (41):
        for num in range (2):
            y_true.append(i)
    RFC = RandomForestClassifier( random_state=42)
    RFC.fit(projected_training_imgs, y)
    print(RFC.score(projected_test_imgs,y_true))
    # projected_test_imgs[0] =projected_test_imgs[0].reshape(-1, 1)
    print(RFC.predict(projected_test_imgs[18].reshape(1,-1 )))
    # clf = SVC().fit(projected_training_imgs,y)
    # y_pred = clf.predict(projected_test_imgs)
    
    # print(accuracy_score(y_true, y_pred))
    return RFC

def show_predicted_image(RFC,mean_image,eigen_vectors):
    test_image_path = "D:/4th year 2nd term/cv/tasks/task5/data/25/244_25.jpg"
    image = cv2.imread(test_image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    subtracted_image = image_gray- mean_image
    subtracted_image = subtracted_image.flatten()
    projected_image = np.dot(eigen_vectors, subtracted_image)
    projected_image =projected_image.reshape(1, -1)
    label = RFC.predict(projected_image)
    print(label[0]+1)
    folder_path = "D:/4th year 2nd term/cv/tasks/task5/data/" + str(label[0]+1)
    for root, _, files in os.walk(folder_path):
        for count, file in enumerate(files) :
            if(count == 0 ):
                img = cv2.imread(os.path.join(root,file))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                break
    plt.imshow(img_gray,cmap="gray")
    plt.show()