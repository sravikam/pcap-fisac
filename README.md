# Image Merging

This was our submission for our college assignment in the subject of Parallel 
Computer Architecture Programming. This project uses the stb_image library 
files to read an image and write an image.

*merge*: Merges two images. The output image is of same size as first image.

*mosaic*: Merges pixels of the same image to form a 32x32 tiled image.

## How To Use

```bash
# Compile the code using the nvcc library
nvcc merge.cu -o merge
nvcc mosaic.cu -o mosaic
```

### Execution

```bash
# To merge images/rr.jpg and images/cuteCat.png to an output image called 
# merge_output.png, type the following code
./merge images/rr.jpg images/cuteCat.png merge_output
```

```bash
# To create a mosaic tiled image of images/rr.jpg
./mosaic images/rr.jpg

# Note: The output file will have the default name of result.png
```