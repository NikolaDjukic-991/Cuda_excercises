#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MIN -1024
#define MAX 1024

#define FALSE 0
#define TRUE 1


#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

// #define BENCH_PRINT


/*----------- using cycle counter ------------*/
__inline__ uint64_t rdtsc()
{
	uint32_t lo, hi;
	/* We cannot use "=A", since this would use %rax on x86_64 */
	__asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
	return (uint64_t)hi << 32 | lo;
}

unsigned long long start_cycles;
#define startCycle() (start_cycles = rdtsc())
#define stopCycle(cycles) (cycles = rdtsc()-start_cycles)

/*--------- using gettimeofday ------------*/

#include <sys/time.h>

struct timeval starttime;
struct timeval endtime;

#define startTime() \
{ \
	gettimeofday(&starttime, 0); \
}
#define stopTime(valusecs) \
{ \
	gettimeofday(&endtime, 0); \
	valusecs = (endtime.tv_sec - starttime.tv_sec) * 1000000 + endtime.tv_usec - starttime.tv_usec; \
}


int SeqPersistence(int x) {
	int sum, y, pers = 0;

	while (x >= 10) {

		sum = 0;
		y = x;
		while (y > 0) {

			sum += y % 10;
			y /= 10;

		}

		x = sum;

		pers++;
	}

	return pers;
}

void SeqArrPersistance(int *in, int *out, int n) {
	int i;
	for (i = 0; i < n; i++) {
		out[i] = SeqPersistence(in[i]);
	}
}

__device__ int Persistance(int x){
	int sum, y, pers = 0;

	while (x >= 10) {

		sum = 0;
		y = x;
		while (y > 0) {

			sum += y % 10;
			y /= 10;

		}

		x = sum;

		pers++;
	}

	return pers;
}


/*
	 0 1 2 3  0 1 2 3  0 1 2 3
	 x x x x  x x x x  x x x x		gridDim.x*blockDim.x = 12
0	 x x x x  x x x x  x x x x
	 x x x x  x x x x  x x x x 
	 x x x x  x x x x  x x x x

	 x x x x  x x x x  x x x x
1	 x x x x  x x x x  x x x x
	 x x x x  x x x x  x x x x
	 x x x x  x x x x  x x x x

*/


__global__ void MatrixPersistance(int* dev_in, int* dev_out, int n, int m){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = idy*gridDim.x*blockDim.x + idx;
	if (index < n*m){
		dev_out[index] = Persistance(dev_in[index]);
	}
}

int main(int argc, char* argv[]) {

	int *in, *dev_in, *dev_out, *out, *cuda_out, i, n,m;
	int test;

	cudaError_t cudaStatus;

	srand(time(NULL));

	if (argc == 3) {
		n = atoi(argv[1]);
		m = atoi(argv[2]);
	}
	else {
		printf("N?");
		scanf("%d", &n);
		printf("M?");
		scanf("%d", &m);
	}

	in = (int*)malloc(m*n * sizeof(int));
	out = (int*)malloc(m*n * sizeof(int));
	cuda_out = (int*)malloc(m*n * sizeof(int));

	for (i = 0; i < m*n; i++) {
		in[i] = abs(rand() / (double)RAND_MAX * (MAX - MIN) + MIN);
	}

	cudaStatus = cudaMalloc((void**)&dev_in, m*n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&dev_out, m*n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(-1);
	}
	
	cudaStatus = cudaMemcpy(dev_in, in, m*n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(-1);
	}
	int numBlocksN = ceil((float)(n) / 32.0);
	int numBlocksM = ceil((float)(m) / 32.0);
	int numThreadsPerBlock = 32;

	dim3 dimGrid(numBlocksN, numBlocksM);
	dim3 dimBlock(numThreadsPerBlock, numThreadsPerBlock);
	double elapsed_time_host, elapsed_time_device;
	startTime();
	MatrixPersistance <<<dimGrid, dimBlock>>>(dev_in, dev_out, n, m);

	cudaStatus = cudaDeviceSynchronize();
	stopTime(elapsed_time_device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(-1);
	}

	cudaMemcpy(cuda_out, dev_out, n*m * sizeof(int), cudaMemcpyDeviceToHost);
	startTime();
	SeqArrPersistance(in, out, n*m);
	stopTime(elapsed_time_host);

	test = TRUE;
	for (i = 0; i < n*m; i++){
		if (out[i] != cuda_out[i]){
			printf("%d\t%d\t%d\n", i, out[i], cuda_out[i]);
			test = FALSE;
			//break;
		}
	}

	if (test){
		printf("TEST PASSED\n");
	}
	else {
		printf("TEST FAILED\n");
	}

	printf("Seq: %ld\tPar: %ld\n", elapsed_time_host, elapsed_time_device);

	cudaFree(dev_in);
	cudaFree(dev_out);
	free(in);
	free(out);
	free(cuda_out);
	system("PAUSE");

	return 0;
}


