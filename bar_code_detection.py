
'''
Created on 22-May-2022

@author: owaish


pseduo code: 
1: conveting into grayscale
2: Highlighting edges
3: Morphological operation to extract relavant regions
4: instance segmentation
5: detection of instances/items

'''

import cv2
import os
from barcode_detect import barcode_detector
import argparse


parser = argparse.ArgumentParser(description='BarCode detector')
parser.add_argument('--img_path', default='./barcode.jpg', type=str, metavar='N',
                    help='path to a image for barcode detection')
parser.add_argument('--outdir', default =  './', type=str, metavar='N',
                    help='path to write the for barcoded image')


args = parser.parse_args()
item_dict = {}
def main():
    filename =  os.path.basename(args.img_path)
    img = cv2.imread(args.img_path)
    img, boxes = barcode_detector(img)
    cv2.imwrite(args.outdir + filename[:-4] + '_detected.png',img)
    print ('Bar code detection is done')
    
if __name__ == '__main__':
    main()
    
    
