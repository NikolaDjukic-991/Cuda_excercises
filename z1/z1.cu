#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

__global__ void ArrPersistance(int* dev_in, int* dev_out, int n){
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n){
		dev_out[idx] = Persistance(dev_in[idx]);
	}
}

int main(int argc, char* argv[]) {

	int *in, *dev_in, *dev_out, *out, *cuda_out, i, n;
	int test;

	cudaError_t cudaStatus;

	srand(time(NULL));

	if (argc == 2) {
		n = atoi(argv[1]);
	}
	else {
		printf("N?");
		scanf("%d", &n);
	}

	in = (int*)malloc(n * sizeof(int));
	out = (int*)malloc(n * sizeof(int));
	cuda_out = (int*)malloc(n * sizeof(int));

	for (i = 0; i < n; i++) {
		in[i] = abs(rand() / (double)RAND_MAX * (MAX - MIN) + MIN);
	}

	cudaStatus = cudaMalloc((void**)&dev_in, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(-1);
	}

	cudaStatus = cudaMalloc((void**)&dev_out, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(-1);
	}

	cudaStatus = cudaMemcpy(dev_in, in, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(-1);
	}
	int numBlocks = ceil((float)(n) / 256.0);
	int numThreadsPerBlock = 256;

	dim3 dimGrid(numBlocks);
	dim3 dimBlock(numThreadsPerBlock);
	startTime();
	ArrPersistance<<<dimGrid, dimBlock>>>(dev_in, dev_out,n);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	stopTime(elapsed_time_device);
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(-1);
	}

	cudaMemcpy(cuda_out, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);

	startTime();
	SeqArrPersistance(in, out, n);
	stopTime(elapsed_time_host);

	test = TRUE;
	for (i = 0; i < n; i++){
		if (out[i] != cuda_out[i]){
			test = FALSE;
		}
	}

	if (test){
		printf("TEST PASSED\n");
	}
	else {
		printf("TEST FAILED\n");
	}

	printf("Seq: %ld\tPar: %ld\n", elapsed_time_host, elapsed_time_device);

	free(in);
	free(out);
	free(cuda_out);
	system("PAUSE");

	return 0;
}


