# Image Merging

This was our submission for our college assignment in the subject of Parallel 
Computer Architecture Programming. This project uses the stb_image library 
files to read an image and then uses CUDA to parallely find average of both 
images at each pixel.

## How To Use

```bash
# Compile the code using the nvcc library
nvcc merge.cu -o merge
```

Note: All files must have the same extension


```bash
./merge image1.png image2.png result
```