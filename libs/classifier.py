# import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
    # clf = SVC().fit(projected_training_imgs,y)
    # y_pred = clf.predict(projected_test_imgs)
    # print(accuracy_score(y_true, y_pred))
    return RFC

def show_predicted_image(RFC,projected_images):
    test_image_path = "D:/4th year 2nd term/cv/tasks/task5/data/4/40_4.jpg"
    person = os.path.basename(test_image_path)
    prefix = person.rpartition('.')[0]
    prefix = prefix.rsplit("_", 1)[-1]
    print("original" ,prefix)
    prefix = int(prefix)
    prefix = int(prefix *2)
    if (prefix % 2 == 1):
        prefix = prefix-1
    # print(prefix)
    label = RFC.predict(projected_images[prefix].reshape(1, -1))
    print("estimated: ",label[0])
    folder_path = "D:/4th year 2nd term/cv/tasks/task5/data/" + str(label[0])
    for root, _, files in os.walk(folder_path):
        for count, file in enumerate(files) :
            if(count == 0 ):
                img = cv2.imread(os.path.join(root,file))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                break
    plt.imshow(img_gray,cmap="gray")
    plt.show()