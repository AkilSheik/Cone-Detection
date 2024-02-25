Below is my answer.png picture: 

![alt text](answer.png "Result")

Methodology: I first applied HSV on the input image to make distinct the various colors. I then applied an orange mask on the input image such that only the cones will appear white and everything else would appear black. I then cleaned this image by eroding and dialating it to remove any noise. Next, I got the contours from this image as a 2D array. There should be one contour for each cone in the picture. Then, I found the centroids of each of the contours  (this would help minimize noise). The array of centroids should be an array of 2D points, where each point represents the location of a cone in the picture. To distinguish the cones on the left from the cones on the right, I used a KMeans clustering algorithim based on the x-coordinate to return two clusters. Finally, for each cluster, I drew the best fit line of the points in the cluster onto the image object and saved the image as answer.png

What did you try and why do you think it did not work: Through trying this appraoch, I was successful in making the two lines go through the cones. An improvement would be to pre-process the centroids before applying the KMeans clustering algorithim. This is because if there were other objects in the picture that had a very similar color as the orange cones, then it the program would accept that object as a contour. This needs to be filtered out through considering otehr aspects of a cone(area for instance). In this case, this did not apply. 

Libraries used: cv2, numpy, sklearn