import cv2
import numpy as np
from operator import itemgetter

def print_octagons(octagons):
    for octagon in octagons:
        print("[%s, %s]" % (str(octagon[0]), str(octagon[1])))

def unify_octagons_x(octagons):
    for left in xrange(1, len(octagons)):
        for right in xrange(left, len(octagons)):
            if abs(octagons[left][0] - octagons[right][0]) < 10:
                octagons[right] = (octagons[left][0], octagons[right][1], octagons[right][2])

def unify_octagons_y(octagons):
    for left in xrange(1, len(octagons)):
        for right in xrange(left, len(octagons)):
            if abs(octagons[left][1] - octagons[right][1]) < 10:
                octagons[right] = (octagons[right][0], octagons[left][1], octagons[right][2])

def export_images(octagons):
    number = 0
    for octagon in octagons:
        number = number + 1
        filename = "out-contours/" + str(number) + ".jpg"
        print("Exporting...[%s]" % (str(number)))
        cv2.imwrite(filename, octagon[2])

img = cv2.imread('test01.jpeg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh,None,iterations = 2)
thresh = cv2.erode(thresh,None)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
number = 0

#template = cv2.imread('search-pattern01.jpeg')

octagons = []
for cnt in contours:
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(hull)
    P = cv2.arcLength(hull,True)

    if ((area != 0) and (13<= P**2/area <= 14)):
        #cv2.drawContours(img,[hull],0,255,3)
        x,y,w,h = cv2.boundingRect(hull)
        roi = img[y:y+h,x:x+w]
        #cv2.imshow(str(number),roi)
        if len(roi) > 89:
            M = cv2.moments(cnt)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            octagons += [(centroid_x, centroid_y, roi)]
            #number = number + 1
            #cv2.imwrite("out-contours/" + str(number) + ".jpg", roi)

print_octagons(octagons)
print(str(len(octagons)))
## Sort by Y (descending)
#octagons.sort(key=itemgetter(1), reverse=True)
## Sort by X (ascending)
#octagons.sort(key=itemgetter(0))

octagons.sort(key=itemgetter(0))
#octagons.sort(key=itemgetter(1))
print('--- sorted by x')
print_octagons(octagons)

print('--- unified by x')
unify_octagons_x(octagons)
print_octagons(octagons)

print('--- unified by y')
octagons.sort(key=itemgetter(1))
unify_octagons_y(octagons)
print_octagons(octagons)

print('--- final result by x')
octagons.sort(key=itemgetter(1))
octagons.sort(key=itemgetter(0))
print_octagons(octagons)

print('--- final result by y')
octagons.sort(key=itemgetter(0))
octagons.sort(key=itemgetter(1))
print_octagons(octagons)
print(str(len(octagons)))

export_images(octagons)
#print('--- sorted by y')
#octagons.sort(key=itemgetter(0), reverse=True)
#octagons.sort(key=itemgetter(1))
#print_octagons(octagons)

#for octagon in octagons:
    

#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()