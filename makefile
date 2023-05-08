all: merge.exe mosaic.exe


merge.exe: merge.cu
	nvcc merge.cu -o merge

mosaic.exe: mosaic.cu
	nvcc mosaic.cu -o mosaic

clean:
	erase *.exe *.exp *.lib