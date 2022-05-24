# Team 15

| Name | sec | BN |
|------|------|----|
|Ahmed Hossam Mohamed Sedky | 1 | 2 |
|Ahmed Mohammed Abdelfatah | 1 | 5 |
|Ehab Wahba Abdelrahman | 1 | 22 |
|Mo'men Maged Mohammed | 2 | 12 |
|Mohanad Alaa Ragab | 2 | 31 |
----
## Libraries versions
* numpy version **1.21.3**
* cv2 version **4.5.4-dev**
* matplotlib version **3.4.2**
* sklearn version **0.24.2**
-----
## Code architecture
* Face Detection 
    * use cv2  to detect faces in the input image 
    ```
        detect_face(image):
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    ```    
* Face Recognition
    * load Images and split it to test images and training images with ratio 80%
    ```
       for root, _ , files in os.walk(folder):
        cnt = 0 
        for file in files:
            img = cv2.imread(os.path.join(root,file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                if (cnt < 8):
                    training_images.append(img_gray)
                    cnt_train += 1
                else:
                    test_images.append(img_gray)
    ```
    * get Mean Image
    ```
       mean_image= np.mean(crop_images,axis=0)
    ```
    * subtract images from the mean image
    ```
    for image in images:
        sub_img = image - mean_image
    ```
    *  get covariance matrix

    ```
    cov_mat = np.dot(substracted_images, substracted_images_Transpose)
        no_images = len(images)
        cov_mat = (1/(no_images -1 )) *np.asarray(cov_mat)
    ```
    * get eigen values and eigen vectors ,Keep all vectors summing up eigen values to 90% and remove the rest
        ```
         eigen_values,eigen_vectors = np.linalg.eig(cov_mat)
        eigen_vectors = np.dot(eigen_vectors, substracted_images)
        tot_eigen_values = np.sum(eigen_values)
        accepted_variance = 0
        cnt = 0
        for eigen_val in eigen_values:
            if ( accepted_variance/ tot_eigen_values < 0.9): 
                accepted_variance += eigen_val
                cnt += 1
        eigen_vectors = eigen_vectors[:cnt , : ] 
        ```
    * Then map all images to new components and it will be in shape (column vector of remaining eigen vectors length)
        ```
            projected_images = np.dot(eigen_vectors, images)
        ```
    * get the accuracy score using Random forest classifier
    * plot ROC & AUC curve
