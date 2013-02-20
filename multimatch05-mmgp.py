import sys
import cv2
import numpy

outbasename = 'out-multimatch05-mmgp/hexagon_%02d.png'

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
value = cv2.copyMakeBorder(255 - value, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)

# Discard small components.
img_clean = numpy.zeros(value.shape, dtype=numpy.uint8)
contours, _ = cv2.findContours(value, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 500:
        cv2.drawContours(img_clean, contours, i, 255, 2)


def closest_pt(a, pt):
    if not len(a):
        return (float('inf'), float('inf'))
    d = a - pt
    return a[numpy.argmin((d * d).sum(1))]

match = cv2.matchTemplate(img_clean, template, cv2.TM_CCORR_NORMED)

# Filter matches.
threshold = 0.8
dist_threshold = twidth / 1.5
loc = numpy.where(match > threshold)
ptlist = numpy.zeros((len(loc[0]), 2), dtype=int)
count = 0
print "%d matches" % len(loc[0])
for pt in zip(*loc[::-1]):
    cpt = closest_pt(ptlist[:count], pt)
    dist = ((cpt[0] - pt[0]) ** 2 + (cpt[1] - pt[1]) ** 2) ** 0.5
    if dist > dist_threshold:
        ptlist[count] = pt
        count += 1

# Adjust points (could do for the x coords too).
ptlist = ptlist[:count]
view = ptlist.ravel().view([('x', int), ('y', int)])
view.sort(order=['y', 'x'])
for i in xrange(1, ptlist.shape[0]):
    prev, curr = ptlist[i - 1], ptlist[i]
    if abs(curr[1] - prev[1]) < 5:
        y = min(curr[1], prev[1])
        curr[1], prev[1] = y, y

# Crop in raster order.
view.sort(order=['y', 'x'])
for i, pt in enumerate(ptlist, start=1):
    cv2.imwrite(outbasename % i,
            img[pt[1]-2:pt[1]+theight-2, pt[0]-2:pt[0]+twidth-2])
    print 'Wrote %s' % (outbasename % i)