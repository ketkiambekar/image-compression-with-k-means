# image-compression-with-k-means

In this project, we will apply the K-means algorithm to lossy image compression, by reducing the number of colors used in an image.

We have a small version (128, 128, 3) and a large version (512, 512, 3) of the same image. The said images are available on the github page and needs to be loaded to files section on the left before executing this notebook.

Our original image is represented by 24-bit colors i.e. each pixel is represented by (2^8, 2^8, 2^8) RGB values.

To compress the image, we will use K-means to reduce the image to k = 16 colors. More speciﬁcally, each pixel in the image is considered a point in the three-dimensional (r, g, b)-space. To compress the image, we will cluster these points in color-space into 16 clusters, and replace each pixel with the closest cluster centroid.

<br><br>

![Results](/images/Result.jpg)
