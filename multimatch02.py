
import sys
import cv2
import numpy
#from scipy.cluster.vq import kmeans,vq
from operator import itemgetter
#import scipy.cluster.hierarchy as hcluster
import numpy.random as random
#import matplotlib.pyplot as plt

def cluster(order, distance, points, threshold):
    ''' Given the output of the options algorithm,
    compute the clusters:

    @param order The order of the points
    @param distance The relative distances of the points
    @param points The actual points
    @param threshold The threshold value to cluster on
    @returns A list of cluster groups
    '''
    clusters = [[]]
    points   = sorted(zip(order, distance, points))
    splits   = ((v > threshold, p) for i,v,p in points)
    for iscluster, point in splits: 
        if iscluster: clusters[-1].append(point)
        elif len(clusters[-1]) > 0: clusters.append([])
    return clusters

    rd, cd, order = optics(points, 4)
    print cluster(order, rd, points, 38.0)

def clean(points):
    pxthreshold = 10
    pythreshold = 10
    for pt1 in points:
        x1 = pt1[0]
        y1 = pt1[1]
        #print('%s, %s' % (str(x1), str(y1)))
        for pt2 in points:
            x2 = pt2[0]
            y2 = pt2[1]
            #print('%s, %s == %s, %s' % (str(x1), str(y1), str(x2), str(y2)))
            if  \
               abs(x1 - x2) <= pxthreshold and \
               abs(y1 - y2) <= pythreshold:
               #((x1 == x2) and \
               #abs(x1 - x2) <= pxthreshold and \
               #abs(y1 - y2) <= pythreshold)):
                #if ((x1 != x2 and y1 != y2) or (x1 == x2) or (y1 == y2)):
                if (x1 == x2 and y1 == y2):
                    pass
                else:
                    #print('deleting 2nd point... %s, %s == %s, %s' % (str(x1), str(y1), str(x2), str(y2)))
                #if (x1 == x2 or y1 == y2) and \
                #   (x1 != x2 and y1 != y2):
                    points.remove(pt2)
                #del(pt2)    

def main(argv):
    if len(argv) >= 3:
        imgfile = argv[1]
        tplfile = argv[2]
        outfile = argv[3]
    else:
    
    #if imgfile is None:
        imgfile = "test01.jpeg"
    #if tplfile is None:
        tplfile = "search-pattern04.jpeg"
    #if outfile is None:
        outfile = "out-multisearch02-07.jpeg"
    
    img = cv2.imread(imgfile)
    template = cv2.imread(tplfile)
    th, tw = template.shape[:2]
    
    result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    #threshold = 0.99
    #threshold = 0.993
    threshold = 0.994
    #threshold = 0.9953
    loc = numpy.where(result >= threshold)
    count = 0
    
    points = zip(*loc[::-1])
    
    #ret,thresh = cv2.threshold(result,127,255,1)
    #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #
    #for cnt in contours:
    #    x,y,w,h = cv2.boundingRect(cnt)
    #    if 10 < w/float(h) or w/float(h) < 0.1:
    #        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    # Sort by Y (descending)
    points.sort(key=itemgetter(1), reverse=True)
    # Sort by X (ascending)
    points.sort(key=itemgetter(0))
    
    print(len(points))
    clean(points)
    print(len(points))
    clean(points)
    print(len(points))
    
    ## Clustering duplicated data
    #thresh = 5
    #clusters = hcluster.fclusterdata(numpy.transpose(points), thresh, criterion="distance")
    #
    ## plotting
    #plt.scatter(*points, c=clusters)
    #plt.axis("equal")
    #title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    #plt.title(title)
    #plt.show()
    
    #clust = cluster(10, 10, points, 10)
    
    for pt in points:
    #    filename = "multisearch-out/image_processed1-"+str(count)+".jpeg"
        re = cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), 0, 2)
        #print(type(pt[0]))
        print('%s, %s' % (str(pt[0]), str(pt[1])))
    #    cv2.imwrite(filename, re)
    
        # http://stackoverflow.com/questions/11396830/converting-opencv-boundingrect-into-numpy-array-with-python
        
        a = result[pt[0]:pt[0] + tw, pt[1]:pt[1] + th]  
        tmp = numpy.array(a, dtype = result.dtype)
        
        #area = [pt, (pt[0] + tw, pt[1] + th)]
        r = cv2.boundingRect(a)
        #filename = "multisearch-out/image_processed2-"+str(count)+".jpeg"
        #cv2.imwrite(filename, img[r[0]:r[0]+r[2], r[1]:r[1]+r[3]])
        #cv2.imwrite(, pt)
        count+=1
    
    print count
    cv2.imwrite(outfile, img)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))