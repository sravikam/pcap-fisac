# Image Merging

This was our submission for our college assignment in the subject of Parallel 
Computer Architecture Programming. This project uses the stb_image library 
files to read an image and then uses CUDA to parallely find average of both 
images at each pixel and outputs it to a new image file.

## How To Use

```bash
# Compile the code using the nvcc library
nvcc merge.cu -o merge

# Execute the file
# ./merge image1.png image2.png name
./merge images/rr.jpg images/cuteCat.png output
```