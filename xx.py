import cv2
import numpy as np
font = cv2.FONT_HERSHEY_COMPLEX #görüntüdeki metni görüntülemek için kullanacağımız yazı tipini tanımlıyoruz
img = cv2.imread('s2.jpeg', 0)
kernel=np.ones((3,3),np.uint8)
dilation=cv2.dilate(img,kernel,iterations=1)
image=cv2.GaussianBlur(dilation,(5,5),0)

dilated_img = cv2.dilate(image, np.ones((7,7), np.uint8)) 
bg_img = cv2.blur(dilated_img,(3,3))
diff_img = 255 - cv2.absdiff(image, bg_img)
_, threshold = cv2.threshold(diff_img, 200, 255, cv2.THRESH_BINARY)

#cv2.imshow("img", img)

floodfill = threshold.copy()
h, w = threshold.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(floodfill, mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(floodfill)

#canny=cv2.Canny(im_floodfill_inv,200,255)
#cv2.imshow("canny", canny)

contours, _ = cv2.findContours(im_floodfill_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #bu komutla siyah beyaz görüntüden tüm şekillerin sınırlanı belirledik

#döngüye sokarız çünkü her şeklin kontur koordinatlarını alırız
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 5)
    
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    
    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), font, 1, (0))
    elif len(approx) == 4:
        cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
    elif len(approx) == 6:
        cv2.putText(img, "Hexagon", (x, y), font, 1, (0))
    else:
        cv2.putText(img, "Circle", (x, y), font, 1, (0))

cv2.imshow("shapes", img)

cv2.imshow("im_floodfill_inv", im_floodfill_inv)
cv2.imshow("bg_img", bg_img)
cv2.imshow("diff_img", diff_img)
cv2.imshow("treshold", threshold)


#cv2.imshow("floodfill", floodfill)
#cv2.imshow("im_floodfill_inv", im_floodfill_inv)






cv2.waitKey(0)
cv2.destroyAllWindows()