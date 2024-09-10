# import the necessary packages
import argparse
import time
import cv2
import os
import numpy as np

###################################################
# Constantes 
side = 1000
img_path = "./examples"
###################################################



def super_resolution_to_img(img_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("models/ESPCN_x4.pb")
    sr.setModel("espcn", 4)
    # load the input image from disk and display its spatial dimensions
    image = cv2.imread(img_path)
    print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))
    upscaled = sr.upsample(image)
    # show the spatial dimensions of the super resolution image
    print("[INFO] w: {}, h: {}".format(upscaled.shape[1], upscaled.shape[0]))
    return upscaled


def resizing_img(img):
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



if __name__ == '__main__':
    print('Inicio de procesamiento de imagenes')
    img_list_to_processed = [file for file in os.listdir(img_path) if file.endswith('png') or file.endswith('jpg')]

    for image in img_list_to_processed:
        file_name = "/".join([img_path, image])
        upscaled = super_resolution_to_img(file_name)
        resized_img = resizing_img(upscaled)
        newImage = concat_image(resized_img)

        path = "imgProcessed/" + image
        print(path)
        cv2.imwrite(path, newImage)

    print('Fin de procesamiento de imagenes')



