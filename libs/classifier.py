# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
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