# image-reducer

A short and simple machine learning project that I built to be able to scale down images to a specific size. The project uses max pooling layers from the Tensorflow library to scale down images and decide which pixels values should be kept when scaling down. This is achieved with the crop_size (m x m) and pool arguments in the reduce_image_folder function. The pixel size of the output image will be the c/p x c/p where c is crop_size and p is the pool.

There is an optional argument for color_restriction that can be set to cluster the colors of the image to a specific number of colors. This is achieved with the KMeans algorithm from the sklearn library. **This significantly highers the computation cost of the function.** Take a look at an example in the backgrounds folder.

### Usage

Use the testing.ipynb file as an example and after importing the correct libraries, play around with the arguments to get a desired image. Make sure to input photos that are close to a square shape for the best results.