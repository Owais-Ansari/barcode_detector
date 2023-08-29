
import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
def locate_item(mask,gray_value = 255):
    X,Y   = np.where(mask==gray_value)
    right,left = max(X),min(X)
    top ,bottom = max(Y),min(Y)
    return left,right,top,bottom#x1,x2,y1,y2

def barcode_detector(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    
    #To find regions in image with high horizontal gradients and low vertical gradients
    
    gradient = cv2.subtract(gradX,gradY)
    gradient = cv2.convertScaleAbs(gradient) 
    
    #blurred  = cv2.blur(gradient , (13,13)) #Blurring to filter out noise in the image
    _,threshold  = cv2.threshold(gradient , 245, 255 , cv2.THRESH_BINARY)   
    #
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7))
    closed = cv2.morphologyEx(threshold , cv2.MORPH_CLOSE , kernel) 
    
    #Morphological series-erode to remove small chunks or noise 
    closed = cv2.erode(closed, (7,7), iterations  = 20)
    
    # closed = cv2.erode(closed, (9,9), iterations  = 18)
    # closed = cv2.erode(closed, (7,7), iterations  = 16) 
    #
    # #Morphological series - dilate to joint the close relevant regions  
    closed = cv2.dilate(closed, (3,11), iterations = 5)
    
    instances = measure.label(closed, background=0)
    
    bboxes = np.unique(instances)
    
    print('num_barcodes or items detected is/are: {}'.format(len(bboxes)-1))
    All_items = {}
    
    for indx,instance in enumerate(bboxes[1:]):
        if instance != 0:
            bbox_points = locate_item(instances,gray_value = instance)
            #rotating the coordinates for drawing the item on it
            start_point = bbox_points[2], bbox_points[0]-25
            end_point   = bbox_points[3], bbox_points[1]
            img = cv2.rectangle(image, start_point, end_point,(0, 255, 0), thickness=3)
            All_items['item_' + str(indx)] = ((bbox_points[2], bbox_points[0] , bbox_points[3], bbox_points[1]))# y1,x1,y2,x2 
        else:
            continue    
    return img,All_items
