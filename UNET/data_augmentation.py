#coding:utf-8
import cv2
import tensorflow as tf
import os
import numpy as np
import csv
images_input_path='./'
labels_input_path='./'


images_output_path='./data_aug'
if not os.path.exists(images_output_path):
    os.makedirs(images_output_path)
#
# labels_output_path = './data/trainlabel_output_crop' + str(crop_rate)
# if not os.path.exists(labels_output_path):
#     os.makedirs(labels_output_path)

input_image_placeholder=tf.placeholder(shape=[None,None,3],dtype=tf.float32)
input_label_placeholder=tf.placeholder(shape=[None,None,3],dtype=tf.float32)

def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated

def read_image(image):

    image = tf.image.random_brightness(image, 0.1)

    #scale
    # image=tf.image.central_crop(image,crop_rate)
    #设置随机的对比度
    image=tf.image.random_contrast(image,lower=0.8,upper=1.2)

    # image = tf.image.resize_images(image, (1200, 1600))
    return image

# images_path_list=[os.path.join(image_input_path,i) for i in os.listdir(image_input_path)]

# print(images_path_list)

if __name__ == '__main__':

    csv_file = open('./train.csv')
    train= csv.reader(csv_file)


    final_iamge=read_image(input_image_placeholder)
    final_label = read_image(input_label_placeholder)
    with tf.Session() as sess:
        # for i in range(2):
        # for j in images_path_list:
        for items in train:

            image_path=os.path.join(images_input_path, items[0])
            label_path = os.path.join(labels_input_path, items[1])

            image=cv2.imread(image_path)
            label = cv2.imread(label_path)
            angle = np.random.randint(low=-180, high=180)

            image = rotate(image, angle)
            label = rotate(label, angle)
            #
            img,label_img=sess.run([final_iamge,final_label],feed_dict={input_image_placeholder:image,input_label_placeholder:label})
            # cv2.imwrite(output_path+'/'+str(i)+j.split('/')[3],img)
            # with open('./train.csv','a') as f:
            img_name = images_output_path + '/' + items[0].split('.')[0].split('/')[1] + '_contrast_rotate'+str(angle)+'.png'
            cv2.imwrite(img_name, img)
            # f.write(img_name+',')
            print('saving img')

            # cv2.imwrite(output_path+'/'+str(i)+j.split('/')[3],img)
            label_name = images_output_path + '/' + items[1].split('.')[0].split('/')[1] + '_contrast_rotate' +str(angle)+'.png'
            cv2.imwrite(label_name, label_img)
            print('saving label')
            # f.write(label_img)
            # f.write('\n')
