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

// Define the kernel that will calculate the average of each tile
__global__ void mosaicImage(uchar* outputImage, uchar* inputImage, int width, int height, int tileSize) {
	int row = blockIdx.y;
	int col = blockIdx.x;

	int startRow = row * tileSize;
	int startCol = col * tileSize;

	int endRow = startRow + tileSize;
	int endCol = startCol + tileSize;

	if (endRow > height) endRow = height;
	if (endCol > width) endCol = width;

	long long total = 0;
	for (int r = startRow; r < endRow; r++) {
		for (int c = startCol; c < endCol; c++) {
			int index = (r * width + c) * CHANNELS + threadIdx.x;
			total += inputImage[index];
		}
	}

	int tilePixels = (endRow - startRow) * (endCol - startCol);
	uchar average = (uchar)(total / tilePixels);

	for (int r = startRow; r < endRow; r++) {
		for (int c = startCol; c < endCol; c++) {
			int index = (r * width + c) * CHANNELS + threadIdx.x;
			outputImage[index] = average;
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

	if(argc < 2 || argc > 3){
		fprintf(stderr, "Usage: ./mosaic inputFile.png [outputFile]");
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

	int smallerDimension = (width < height)? width:height;
	int tileSize = smallerDimension>>5;

	int numTilesX = (width + tileSize - 1) / tileSize;
	int numTilesY = (height + tileSize - 1) / tileSize;

	uchar *dev_dest, *dev_src;

	cudaMalloc(&dev_dest, width * height * CHANNELS);
	cudaMalloc(&dev_src, width * height * CHANNELS);

	cudaMemcpy(dev_src, image, width * height * CHANNELS, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(CHANNELS);
	dim3 numBlocks(numTilesX, numTilesY);
	mosaicImage<<<numBlocks, threadsPerBlock>>>(dev_dest, dev_src, width, height, tileSize);

	uchar *result = (uchar*)malloc(width * height * CHANNELS);
	cudaMemcpy(result, dev_dest, width * height * CHANNELS, cudaMemcpyDeviceToHost);

	stbi_write_png(outputFileName, width, height, CHANNELS, result, width * CHANNELS);

	free(result);
	free(image);
	free(outputFileName);
	cudaFree(dev_dest);
	cudaFree(dev_src);
	return 0;
}