import sys
import cv2
import numpy

outbasename = 'out-multisearch04/hexagon_%02d.png'

img = cv2.imread(sys.argv[1])
template = cv2.cvtColor(cv2.imread(sys.argv[2]), cv2.COLOR_BGR2GRAY)
theight, twidth = template.shape[:2]

# Binarize the input based on the saturation and value.
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
saturation = hsv[:,:,1]
value = hsv[:,:,2]
value[saturation > 35] = 255
value = cv2.threshold(value, 0, 255, cv2.THRESH_OTSU)[1]
# Pad the image.
value = 255 - cv2.copyMakeBorder(value, 3,3,3,3, cv2.BORDER_CONSTANT, value=255)

# Discard small components.
img_clean = numpy.zeros(value.shape, dtype=numpy.uint8)
contours, _ = cv2.findContours(value, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 500:
        cv2.drawContours(img_clean, contours, i, 255, 2)


match = cv2.matchTemplate(255-img_clean, template, cv2.TM_CCORR_NORMED)

# Filter matches.
threshold = 0.97
#threshold = 0.99
dist_threshold = twidth / 1.5
loc = numpy.where(match > threshold)
ptlist = []
for pt in zip(*loc[::-1]):    
    mindist = float('inf')
    for p in ptlist:
        dist = ((p[0] - pt[0]) ** 2 + (p[1] - pt[1]) ** 2) ** 0.5
        if dist < mindist:
            mindist = dist
    if mindist > dist_threshold:
        # Found a new hexagon.
        ptlist.append(list(pt))

# Adjust points.
ptlist = sorted(ptlist, key=lambda x: x[1])
for i in xrange(1, len(ptlist)):
    prev = ptlist[i - 1]
    curr = ptlist[i]
    if abs(curr[1] - prev[1]) < 5:
        y = min(curr[1], prev[1])
        curr[1] = y
        prev[1] = y

# Crop in raster order.
ptlist = sorted(ptlist, cmp=lambda x, y: cmp(x[1], y[1]) or cmp(x[0], y[0]))
for i, pt in enumerate(ptlist, start=1):
    cv2.imwrite(outbasename % i,
            img[pt[1]+1:pt[1]+theight-2, pt[0]+1:pt[0]+twidth-2])
    #re = cv2.rectangle(img, pt, (pt[0]+1+twidth-2, pt[1]+theight-2), 0, 2)
    #re = cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), 0, 2)
    print 'Wrote %s' % (outbasename % i)