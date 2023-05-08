#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

__global__ void merge_images(uint8_t* dest, uint8_t* src1, int width1, 
	int height1, uint8_t* src2, int width2, int height2){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width1 && y < height1) {
		int index1 = y * width1 + x;
		int index2 = (y * height2 / height1) * width2 + (x * width2 / width1);

	/*
		dest[index1 * 4 + 0] = (src1[index1 * 4 + 0] + src2[index2 * 4 + 0]) / 2;
		dest[index1 * 4 + 1] = (src1[index1 * 4 + 1] + src2[index2 * 4 + 1]) / 2;
		dest[index1 * 4 + 2] = (src1[index1 * 4 + 2] + src2[index2 * 4 + 2]) / 2;
		dest[index1 * 4 + 3] = (src1[index1 * 4 + 3] + src2[index2 * 4 + 3]) / 2;
	*/
		dest[index1 * 3 + 0] = (src1[index1 * 3 + 0] + src2[index2 * 3 + 0]) / 2;
		dest[index1 * 3 + 1] = (src1[index1 * 3 + 1] + src2[index2 * 3 + 1]) / 2;
		dest[index1 * 3 + 2] = (src1[index1 * 3 + 2] + src2[index2 * 3 + 2]) / 2;
	}
}

int main(int argc, char* argv[]){

	if(argc != 3){
		fprintf(stderr, "Usage: ./merge inputFile1.png inputFile2.png");
		exit(1);
	}

	int width1, height1, channels1;

	//uint8_t *image1 = stbi_load(argv[1], &width1, &height1, &channels1, STBI_rgb_alpha);
	uint8_t *image1 = stbi_load(argv[1], &width1, &height1, &channels1, STBI_rgb);

	int width2, height2, channels2;

	//uint8_t *image2 = stbi_load(argv[2], &width2, &height2, &channels2, STBI_rgb_alpha);
	uint8_t *image2 = stbi_load(argv[2], &width2, &height2, &channels2, STBI_rgb);

	// Allocate memory on the device
	uint8_t *dev_dest, *dev_src1, *dev_src2;

	/*
	cudaMalloc((void**)&dev_dest, width1 * height1 * 4);
	cudaMalloc((void**)&dev_src1, width1 * height1 * 4);
	cudaMalloc((void**)&dev_src2, width2 * height2 * 4);
	*/
	cudaMalloc((void**)&dev_dest, width1 * height1 * 3);
	cudaMalloc((void**)&dev_src1, width1 * height1 * 3);
	cudaMalloc((void**)&dev_src2, width2 * height2 * 4);

	// Copy the input images to the device
	//cudaMemcpy(dev_src1, image1, width1 * height1 * 4, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_src2, image2, width2 * height2 * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_src1, image1, width1 * height1 * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_src2, image2, width2 * height2 * 3, cudaMemcpyHostToDevice);

	// Define block and grid sizes
	dim3 blockSize(16, 16);
	dim3 gridSize((width1 + blockSize.x - 1) / blockSize.x, (height1 + blockSize.y - 1) / blockSize.y);

	// Call the kernel to merge the images
	merge_images<<<gridSize, blockSize>>>(dev_dest, dev_src1, width1, height1, dev_src2, width2, height2);

	// Copy the result back to the host
	//uint8_t *result = (uint8_t*)malloc(width1 * height1 * 4);
	//cudaMemcpy(result, dev_dest, width1 * height1 * 4, cudaMemcpyDeviceToHost);
	uint8_t *result = (uint8_t*)malloc(width1 * height1 * 3);
	cudaMemcpy(result, dev_dest, width1 * height1 * 3, cudaMemcpyDeviceToHost);

	// Save the result to disk
	//stbi_write_png("result.png", width1, height1, 4, result, width1 * 4);
	stbi_write_png("result.jpg", width1, height1, 3, result, width1 * 3);

	// Free memory on the host and device
	free(result);
	stbi_image_free(image1);
	stbi_image_free(image2);
	cudaFree(dev_dest);
	cudaFree(dev_src1);
	cudaFree(dev_src2);
	return 0;
}