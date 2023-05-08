all: merge mosaic


merge: merge.cu
	nvcc merge.cu -o merge

mosaic: mosaic.cu
	nvcc mosaic.cu -o mosaic

clean:
	rm merge mosaic