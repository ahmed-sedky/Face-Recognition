import cv2

def detect_face(image):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in  faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_img = image[ y:y+h, x:x+w]
    cv2.imshow("face detection" ,image)
    cv2.waitKey()
    if (len(faces) != 0 ):
        return crop_img ,len(faces)
    else:
        return image,len(faces)

