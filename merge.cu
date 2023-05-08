#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHANNELS 3

typedef unsigned char uchar;

__global__ void merge_images(uchar* dest, uchar* src1, int width1, 
	int height1, uchar* src2, int width2, int height2){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width1 && y < height1) {
		int index1 = (y * width1 + x)*CHANNELS;
		int index2 = ((y * height2 / height1) * width2 + (x * width2 / width1))*CHANNELS;

		for(int j=0; j<CHANNELS; ++j){
			dest[index1 + j] = (src1[index1 + j] + src2[index2 + j]) / 2;
		}
	}
}

// Returns length if valid or 0 if invalid
int isValidFilename(const char* filename){
	const char* invalidChars = "\\/:*?\"<>|";
	int len = strlen(filename);
	for (int i = 0; i < len; ++i) {
		if (strchr(invalidChars, filename[i]) != NULL) {
			return 0;
		}
	}
	return len;
}

int main(int argc, char* argv[]){

	if(argc < 3 || argc > 4){
		fprintf(stderr, "Usage: ./merge inputFile1.png inputFile2.png [outputFile]");
		exit(1);
	}

	char* outputFileName;
	
	if(argc == 4){
		if(strrchr(argv[3], '.')){
			fprintf(stderr, "Error: Output file should not be given an extension\n");
			exit(1);
		}
		int value = isValidFilename(argv[3]);
		if(value){
			outputFileName = (char*)calloc(value+5, sizeof*outputFileName);
			strcpy(outputFileName, argv[3]);
			strcat(outputFileName, ".png");
		} else{
			fprintf(stderr, "Error: Invalid output file name");
		}
	} else{
		outputFileName = (char*)calloc(11, sizeof*outputFileName);
		strcpy(outputFileName, "result.png");
	}
	

	int width1, height1;
	uchar *image1 = stbi_load(argv[1], &width1, &height1, NULL, STBI_rgb);
	if(!image1){
		fprintf(stderr, "Error: %s\n", stbi_failure_reason());
		exit(1);
	}

	int width2, height2;
	uchar *image2 = stbi_load(argv[2], &width2, &height2, NULL, STBI_rgb);
	if(!image2){
		fprintf(stderr, "Error: %s\n", stbi_failure_reason());
		exit(1);
	}

	// Allocate memory on the device
	uchar *dev_dest, *dev_src1, *dev_src2;

	cudaMalloc((void**)&dev_dest, width1 * height1 * CHANNELS);
	cudaMalloc((void**)&dev_src1, width1 * height1 * CHANNELS);
	cudaMalloc((void**)&dev_src2, width2 * height2 * CHANNELS);

	// Copy the input images to the device
	cudaMemcpy(dev_src1, image1, width1 * height1 * CHANNELS, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_src2, image2, width2 * height2 * CHANNELS, cudaMemcpyHostToDevice);

	// Define block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((width1 + blockSize.x - 1) / blockSize.x, (height1 + blockSize.y - 1) / blockSize.y);

	// Call the kernel to merge the images
	merge_images<<<gridSize, blockSize>>>(dev_dest, dev_src1, width1, height1, dev_src2, width2, height2);

	// Copy the result back to the host
	uchar *result = (uchar*)malloc(width1 * height1 * CHANNELS);
	cudaMemcpy(result, dev_dest, width1 * height1 * CHANNELS, cudaMemcpyDeviceToHost);

	// Save the result to disk
	stbi_write_png(outputFileName, width1, height1, CHANNELS, result, width1 * CHANNELS);

	// Free memory on the host and device
	free(result);
	free(image1);
	free(image2);
	free(outputFileName);
	cudaFree(dev_dest);
	cudaFree(dev_src1);
	cudaFree(dev_src2);
	return 0;
}