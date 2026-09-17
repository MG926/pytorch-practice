import cv2
import matplotlib.pyplot as plt
def imageSplit(img_path,show):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    _,binary_img = cv2.threshold(
        img,
        127,
        255,
        cv2.THRESH_BINARY_INV)
    conters,_ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    conters = sort_conters(conters)
    digit_images = []
    for conter in conters:
        x,y,w,h = cv2.boundingRect(conter)
        if(h>20):
            padding = 30
            img_h,img_w = binary_img.shape
            x1 = max(x-padding,0)
            x2 = min(x+w+padding,img_w)
            y1 = max(y-padding,0)
            y2 = min(y+padding+h,img_h)
            digit_img = binary_img[y1:y2,x1:x2]
            digit_images.append(digit_img)
    if(show):
        plt.figure()
        n = len(digit_images)
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.tight_layout()
            plt.imshow(digit_images[i])
        plt.show()
    return digit_images

def sort_conters(conters):
    conters_lists = list(conters)
    conters_lists.sort(key= lambda c: cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2]/2)
    return conters_lists