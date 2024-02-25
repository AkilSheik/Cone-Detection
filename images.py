import cv2
import numpy as np
from sklearn.cluster import KMeans

#Takes the cv2 image object, applies a HSV mask
#over it(to contrast the colors) and applies an 
#orange mask to detect the cones. Then 
#dialates and erodes the image. Returns a black
#and white image where it is white only where
#the cones are. 
def get_masked_image(image):
    #Applies HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Applies the orange mask to detect the cones
    lower_boundary = np.array([179, 210, 165])
    upper_boundary = np.array([255, 255, 255])
    mask = cv2.inRange(hsv_img, lower_boundary, upper_boundary)

    #Dialates and erodes the image to fill in the areas of the 
    #image containing cones and to remove other noises
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

#Given an image (the masked image from above), returns a 2D arary containing 
#the contours. Each array in the larger array represents a contour while the 
#elements withiin the array are 2D points in the contour
def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#For each contour, calculates the centroid of all of the points within the contour and 
#returns a 1D array of these centroids. Does this by calculating the moments. Each
#centroid should represent the location of one orange cone. 
def get_centroids(contours):
    centroids = []
    for x in range (0, len(contours)):
        moment = cv2.moments(contours[x]) #Calculates the moments of each contour
        if moment['m00'] != 0:            #Then calculates centroid of each contour and appends
                                          #it to the array
            centroid = (int(moment["m10"]/moment["m00"]), int(moment["m01"]/moment["m00"]))
            centroids.append(centroid)
    return centroids

#Uses KMeans clustering to split these centroids into two regions based on their 
#x coordinate. This will differentiate the cones in the left row from the cones
#on the right row. Returns a dictionary of centroids belonging to each cluster
def get_clusters(centroids, cluster_num):
    centroids_x = [centroid[0] for centroid in centroids] #Extracts the x coordinate
    #Creates the cluster labels for each centroid(left or right represented by 0 or 1)
    cluster_labels = KMeans(n_clusters=cluster_num, random_state=0).fit(np.array(centroids_x).reshape(-1, 1)).labels_
    #Creates a dictionary storing two numpy arrays(one for each cluster). Each numpy array consists of an array to
    #store the centroids (points of the cones) that belong to that cluster  
    clusters = {x: np.array([]).reshape(0, 2) for x in range(cluster_num)}

    #Goes through each centroid and adds it to the cluster dictionary depending on its label 
    for x, centroid in enumerate(centroids):
        label = cluster_labels[x]
        clusters[label] = np.vstack([clusters[label], centroid])
    return clusters

#Given the original image object and the clusters dictionary containing the centroids(cone points)
#belonging to each cluster, draws the best_fit lines through the image for each cluster. In this 
#case, two lines should be drawn(since each row of cones represents a centroid).
def add_lines(clusters, image):
    #Iterates through all clusters
    for cluster in clusters.values():
        #Gets the best fit line for each cluster and does calculations
        #to fit the line within the image
        [dir_x,dir_y,x,y] = cv2.fitLine(cluster, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = dir_y/dir_x #Slope
        max_height = image.shape[1]
        y_int = (-x * slope) + y #Y intercept
        right_y = ((max_height - x) * slope) + y
        pt1 = (0, int(y_int))
        pt2 = (int(max_height - 1), int(right_y))
        #Displays the line for each centroid onto the image
        cv2.line(image, pt1, pt2, (0, 0, 255), 2)

#Displays the image
def display_image(image):
    cv2.imshow('Detected Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Saves the image to answer.png
def save_image(image):
    cv2.imwrite('answer.png', image)


#Draws the two lines for the two rows of cones
def main():
    CLUSTER_NUM = 2 #Defines number of rows of cones(2)
    image = cv2.imread('red.png') #Reads in the image
    #Gets the masked image highlighting only the cones 
    masked_img = get_masked_image(image) 
    #Gets contours and centroids of image
    contours = get_contours(masked_img)
    centroids = get_centroids(contours)
    #Creates two clusters to map the cones to either 
    #the left row or the right row. 
    clusters = get_clusters(centroids, CLUSTER_NUM)
    add_lines(clusters, image) #Draws the best fit line for each row
    save_image(image) #Saves image to answer.png

main()


