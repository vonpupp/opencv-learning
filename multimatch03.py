import sys
import cv2
import numpy

outbasename = 'out-multisearch03/hexagon_%d.png'

img = cv2.imread(sys.argv[1])
template = cv2.imread(sys.argv[2])
theight, twidth = template.shape[:2]

# Binarize the inputs.
l = [img, template]
for i in xrange(len(l)):
    _, l[i] = cv2.threshold(cv2.cvtColor(l[i], cv2.COLOR_BGR2GRAY),
            0, 255, cv2.THRESH_OTSU)
# Pad the image.
l[0] = 255 - cv2.copyMakeBorder(l[0], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)

# Discard small components.
img_clean = numpy.zeros(l[0].shape, dtype=numpy.uint8)
contours, _ = cv2.findContours(l[0], cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 500:
        cv2.drawContours(img_clean, contours, i, 255, 2)


match = cv2.matchTemplate(255-img_clean, l[1], cv2.TM_CCORR_NORMED)

# Filter matches.
threshold = 0.95
#threshold = 0.99
dist_threshold = 10
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
    else:
        continue

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
    print 'Wrote %s' % (outbasename % i)