all:main

main:main.cu
	nvcc -o $@ -O2 $< -arch=sm_80
