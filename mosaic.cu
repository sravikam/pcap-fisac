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

__global__ uchar mosaicImage(uchar* dest, uchar* src, int width, int height){
	
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

	if(argc < 2 || argc > 3){
		fprintf(stderr, "Usage: ./merge inputFile.png [outputFile]");
		exit(1);
	}

	char* outputFileName;
	
	if(argc == 3){
		if(strrchr(argv[2], '.')){
			fprintf(stderr, "Error: Output file should not be given an extension\n");
			exit(1);
		}
		int value = isValidFilename(argv[2]);
		if(value){
			outputFileName = (char*)calloc(value+5, sizeof*outputFileName);
			strcpy(outputFileName, argv[2]);
			strcat(outputFileName, ".png");
		} else{
			fprintf(stderr, "Error: Invalid output file name");
		}
	} else{
		outputFileName = (char*)calloc(11, sizeof*outputFileName);
		strcpy(outputFileName, "result.png");
	}
	

	int width, height;
	uchar *image = stbi_load(argv[1], &width, &height, NULL, STBI_rgb);
	if(!image){
		fprintf(stderr, "Error: %s\n", stbi_failure_reason());
		exit(1);
	}

	// Allocate memory on the device
	uchar *dev_dest, *dev_src;

	cudaMalloc((void**)&dev_dest, width * height * CHANNELS);
	cudaMalloc((void**)&dev_src, width * height * CHANNELS);

	// Copy the input images to the device
	cudaMemcpy(dev_src, image, width * height * CHANNELS, cudaMemcpyHostToDevice);

	// Define block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	// Call the kernel to merge the images
	mosaicImage<<<gridSize, blockSize>>>(dev_dest, dev_src, width, height);

	// Copy the result back to the host
	uchar *result = (uchar*)malloc(width * height * CHANNELS);
	cudaMemcpy(result, dev_dest, width * height * CHANNELS, cudaMemcpyDeviceToHost);

	// Save the result to disk
	stbi_write_png(outputFileName, width, height, CHANNELS, result, width * CHANNELS);

	// Free memory on the host and device
	free(result);
	free(image);
	free(outputFileName);
	cudaFree(dev_dest);
	cudaFree(dev_src);
	return 0;
}