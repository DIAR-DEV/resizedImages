# import the necessary packages
import argparse
import time
import cv2
import os
import numpy as np


side = 1000

def super_resolution():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to super resolution model")
    ap.add_argument("-i", "--image", required=True,
        help="path to input image we want to increase resolution of")
    args = vars(ap.parse_args())


    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(args["model"])

    sr.setModel("espcn", 4)

    # load the input image from disk and display its spatial dimensions
    image = cv2.imread(args["image"])
    # print(image)
    print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
    # use the super resolution model to upscale the image, timing how
    # long it takes
    start = time.time()
    upscaled = sr.upsample(image)
    end = time.time()
    print("[INFO] super resolution took {:.6f} seconds".format(
        end - start))
    # show the spatial dimensions of the super resolution image
    print("[INFO] w: {}, h: {}".format(upscaled.shape[1],
    upscaled.shape[0]))

    return upscaled, args["image"]


def resizing(img):
    if img.shape[0] < img.shape[1]: #Imagen ancha
        r = side/img.shape[1]
        dim = (side, int(img.shape[0] * r))
        return cv2.resize(img, dim)
    
    elif img.shape[0] > img.shape[1]: #Imagen larga
        r = side/img.shape[0]
        dim = (int(img.shape[1] * r), side)
        return cv2.resize(img, dim)
    
    else:
        return cv2.resize(img, (side, side))
    

def concat_image(upscaled):
    blank_image = np.zeros((side, side, 3), np.uint8)
    blank_image[:,0:side//2] = (255,255,255)      # (B, G, R)
    blank_image[:,side//2:side] = (255,255,255)

    xVal = int((side - upscaled.shape[0])/2)
    yVal = int((side - upscaled.shape[1])/2)

    blank_image[xVal:upscaled.shape[0]+xVal, yVal:upscaled.shape[1]+yVal] = upscaled
    
    return blank_image



upscaled, image_name = super_resolution()
upscaled=resizing(upscaled)
newImage = concat_image(upscaled)


image_name = image_name.replace("examples/", "")
path = "imgProcessed/" + image_name

cv2.imwrite(path, newImage)
# cv2.imshow("Concat image", newImage)

cv2.waitKey(0)



