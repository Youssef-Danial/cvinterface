from django.shortcuts import render
from main.filehandler import *
from django.http import HttpResponseRedirect, HttpResponse
from .models import postfile
from keras.models import load_model
import tensorflow as tf
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2
# Create your views here.
def mainpage(request):
    files = receive_files(request, state="post")
    data = {
        "formfile":files
    }
    if type(files) is list:
        # doing the segmentation on the files here and then rendering the results to the user on another page
        counter = 0
        result_path_list = []
        for file in files:
            result_path_list.append(segment(file, counter))
            result_path_list.append(segmentO(file, counter))
            result_path_list.append(segment_hsv(file, counter))
            counter +=1
        data = {
            "resultlist":result_path_list,
        }
        return render(request, "main/result.html", data)
    return render(request, "main/homepage.html", data)

def segment(file, num):
    # the path of the file file.file_name
    # loading model  
    #drive_path = "disease_segment_unet_augO35x5_90.5_loss_ 0.2293.h5"
    model = tf.keras.models.load_model("main/disease_segment_unet_augO35x5_90.5_loss_0.2293.h5")
    # reading the image 
    image_path = file.file_name
    image_testt = imread(image_path)
    # resizing the image 
    image_test = resize(image_testt, (128,128), mode="constant", preserve_range = True)
    image_testview = resize(image_testt, (128,128))
    print(image_test.shape)
    image_x = np.zeros((1,128, 128, 3),dtype=np.uint8)
    image_x[0] = image_test[:,:,:3]
    print(image_test.shape)
    preds_train = model.predict(image_x[:], verbose=1)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    mask = recreate_image(preds_train_t, 0)
    res = cv2.bitwise_and(image_x[0], image_x[0], mask=mask)
    fig = plt.figure(figsize=(20,10))
    fig.suptitle("Image Semgentation using modified U-Net")
    fig.add_subplot(1, 3, 1)
    # showing the image before range compression
    plt.imshow(image_testview)
    plt.title("Before")
    # showing the image after range compression
    fig.add_subplot(1, 3, 2)
    plt.imshow(recreate_image(preds_train_t, 0), cmap="gray")
    plt.title("Mask")
    fig.add_subplot(1, 3, 3)
    plt.imshow(res)#get_image(preds_train_t, 0, image_testt)
    plt.title("After")
    result_path = f"main/static/result{num}.png"
    plt.savefig(f"main/static/result{num}.png")
    return result_path

def segmentO(file, num):
    # the path of the file file.file_name
    # loading model  
    #drive_path = "disease_segment_unet_augO35x5_90.5_loss_0.2293.h5"
    model = tf.keras.models.load_model("main/best_mode_acc91.5_loss0.204.h5")
    # reading the image 
    image_path = file.file_name
    image_testt = imread(image_path)
    # resizing the image 
    image_test = resize(image_testt, (128,128), mode="constant", preserve_range = True)
    image_testview = resize(image_testt, (128,128))
    print(image_test.shape)
    image_x = np.zeros((1,128, 128, 3),dtype=np.uint8)
    image_x[0] = image_test[:,:,:3]
    # resizing the image 
    image_test = resize(image_testt, (256,256), mode="constant", preserve_range = True)
    image_testview = resize(image_testt, (256,256))
    print(image_test.shape)
    image_x2 = np.zeros((1,256, 256, 3),dtype=np.uint8)
    image_x2[0] = image_test[:,:,:3]
    print(image_test.shape)
    preds_train = model.predict([image_x[:],image_x2[:]], verbose=1)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    mask = recreate_image(preds_train_t, 0)
    res = cv2.bitwise_and(image_x[0], image_x[0], mask=mask)
    fig = plt.figure(figsize=(20,10))
    fig.suptitle("Image Semgentation using modified U-Net 3x3 down 5x5 up with batchnormalization")
    fig.add_subplot(1, 3, 1)
    # showing the image before range compression
    plt.imshow(image_testview)
    plt.title("Before")
    # showing the image after range compression
    fig.add_subplot(1, 3, 2)
    plt.imshow(recreate_image(preds_train_t, 0), cmap="gray")
    plt.title("Mask")
    fig.add_subplot(1, 3, 3)
    plt.imshow(res)#get_image(preds_train_t, 0, image_testt)
    plt.title("After")
    result_path = f"main/static/fresult{num}.png"
    plt.savefig(f"main/static/fresult{num}.png")
    return result_path

def segment_hsv(file,num):
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    low_green_test = np.array([36, 50, 70])
    high_green_test = np.array([89, 255, 255])
    # getting the path of the image
    image_path = file.file_name
    image = cv2.imread(image_path)
    imageclusters = cluster_image(image)
    image,imagemask = grabcut_image(imageclusters,image)
    # convert BGR to HSV
    image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
    #image = resize(image, (128,128), mode="constant", preserve_range = True)
    # green [[89, 255, 255], [36, 50, 70]]
    # convert BGR to HSV
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, low_green_test, high_green_test)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # inverse mask
    maskr = 255-mask
    diease_mask = get_disease_mask(imagemask, maskr)
    #segmented_image = cv2.bitwise_and(image, image, mask=mask)
    segmented_image = cv2.bitwise_and(image, image, mask=maskr)
    print(segmented_image.shape)
    print(f"mask {mask.shape}")
    fig = plt.figure(figsize=(20,10))
    fig.suptitle("Image Semgentation using cluster->graphcut->HSV color range")
    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Before")
    fig.add_subplot(1, 3, 2)
    plt.imshow(diease_mask, cmap="gray")
    plt.title("Mask")
    fig.add_subplot(1, 3, 3)
    plt.imshow(segmented_image)
    plt.title("After")
    result_path = f"main/static/hsvresult{num}.png"
    plt.savefig(f"main/static/hsvresult{num}.png")
    return result_path

def recreate_image(instancelist, image_num):
  test = np.zeros((128,128), dtype=np.uint8)
  for x in range(128):
    for y in range(128):
      if instancelist[image_num][x,y] == True:
          test[x,y] = 255
      else:
          test[x,y] = 0
  return test



# clustering function
def cluster_image(image):
    #image = cv2.imread("apple.JPG")
    # image reshaping to make it a 1 dimension array with 3 channels
    # resizing the image to save time
    dim = (128,128)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image_flat = resized .reshape((-1,3))
    # now converting the datatype of the image to float32 to be compatiable with opencv k-means clustering
    image_flat = np.float32(image_flat)
    # setting a criteria for the k-means algorithm to know when to stop
    # max number of iterations and number and epslon is reached 10, 1.0
    criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    # number of clusters
    k = 6
    attempts = 10
    # doing the k-mean clustering on the image
    # center of each cluster and label for each pixel ret the distance between each pixel and its center
    # pp something called careful seeding
    ret, label, center = cv2.kmeans(image_flat,k, None, criteria, attempts,cv2.KMEANS_PP_CENTERS)
    # changing the center to unit so we can plot it
    center = np.uint8(center)
    # 
    result = center[label.flatten()]
    # reshaping the result
    result_image = result.reshape((128,128,3))
    return result_image
def grabcut_image(image, imagereal):
    # image here represent the image clustered
    #image = cv2.imread("segmentedimage.jpg")
    # i can use image resizing here later if i wanted to make sure that the image is in small shape
    # resizing the image to save time
    dim = (128,128)
    imagereal = cv2.resize(imagereal, dim, interpolation = cv2.INTER_AREA)
    # mask
    mask = np.zeros(image.shape[:2], np.uint8)
    # bgd and fgd models
    bgdmodel = np.zeros((1,65), np.float64)
    fgdmodel = np.zeros((1,65), np.float64)
    # in my case the rect would be all the image because it will be already been
    rect = (0,0,120,120)
    # doing the grap cut using opencv
    cv2.grabCut(image, mask,rect, bgdmodel, fgdmodel, 5, cv2.GC_INIT_WITH_RECT)
    print("grabcut")
    mask2 = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    image_result = imagereal*mask2[:,:,np.newaxis]
    return image_result, mask2

def get_disease_mask(leafmask, greenmask):
    greenmask = greenmask
    diseasemask =  cv2.bitwise_and(greenmask,leafmask)
    return diseasemask

def get_rect(image): # image size should be 128
    pass