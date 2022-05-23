from libs import load_imgs,face_detect,mean_img,sub_img,cov_matrix,eigen,project,classifier
import cv2
import numpy as np

folder = "D:/4th year 2nd term/cv/tasks/task5/data" 
training_images,test_images = load_imgs.load_images_from_folder(folder)

mean_image =  mean_img.get_mean_img (training_images)
substracted_training_images = sub_img.get_sub_images(mean_image,training_images)
substracted_test_images = sub_img.get_sub_images(mean_image,test_images)

cov_mat = cov_matrix.get_cov_mat(substracted_training_images,training_images)
eigen_vectors = eigen.get_eigen(cov_mat,substracted_training_images)
projected_training_imgs = project.get_projection(eigen_vectors,substracted_training_images)
projected_test_imgs = project.get_projection(eigen_vectors,substracted_test_images)
classifier.classify(projected_training_imgs,projected_test_imgs)
