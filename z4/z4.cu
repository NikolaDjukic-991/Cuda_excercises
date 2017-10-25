#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


#include <unistd.h>
#include <stdint.h>


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


//#include "timer.h"

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

//#define BENCH_PRINT

int rows, cols;
int* h_data;
int** h_wall;
int* h_result;

int* h_result_cuda;
int* d_result;
int* d_data;
int* d_nextIteration;

#define M_SEED 9
#define TRUE 1
#define FALSE 0

void
init(int argc, char** argv)
{
	if (argc == 3){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	}
	else{
		printf("Usage: pathfiner width num_of_steps\n");
		exit(0);
	}
	h_data = new int[rows*cols];
	h_wall = new int*[rows];
	for (int n = 0; n<rows; n++)
		h_wall[n] = h_data + cols*n;
	h_result = new int[cols];

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			h_wall[i][j] = rand() % 10;
		}
	}
	for (int j = 0; j < cols; j++)
		h_result[j] = h_wall[0][j];
#ifdef BENCH_PRINT
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%d ", wall[i][j]);
		}
		printf("\n");
	}
#endif
}

void initCuda(){
	cudaMalloc((void**)&d_data, sizeof(int)*cols*rows);
	cudaMalloc((void**)&d_result, sizeof(int)*cols);
	cudaMalloc((void**)&d_nextIteration, sizeof(int)*cols);

	h_result_cuda = new int[cols];

	cudaMemcpy(d_data, h_data, sizeof(int)*rows*cols, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, h_wall[0], sizeof(int)*cols, cudaMemcpyHostToDevice);
}


#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


// -------- -------- -------- -------- -------- --------
// --------

#define BLOCKSIZE 256

__global__ void pathfinder_iteration(int* d_data, int* d_result, int rows, int cols){
	__shared__ int s_data[BLOCKSIZE * 2];
	int *temp, *dst, *src;
	int* min;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// init shared
	src = s_data;
	min = s_data + BLOCKSIZE;

	// source is the result of the prev iteration
	if (idx < cols){
		src[threadIdx.x] = d_result[idx];										// 7 2 5 4 7 2 5 1


		min[threadIdx.x] = src[threadIdx.x];
		if (idx > 0){
			if (threadIdx.x != 0)
				min[threadIdx.x] = MIN(min[threadIdx.x], src[threadIdx.x - 1]);
			else
				min[threadIdx.x] = MIN(min[threadIdx.x], d_result[idx - 1]);
		}
		if (idx < cols - 1){
			if (threadIdx.x + 1 != blockDim.x)
				min[threadIdx.x] = MIN(min[threadIdx.x], src[threadIdx.x + 1]);
			else
				min[threadIdx.x] = MIN(min[threadIdx.x], d_result[idx + 1]);
		}

		d_result[idx] = d_data[cols + idx] + min[threadIdx.x];					// 5 1 2 5 1 2 5 1																				// 7 3 4 9 3 4 6 2
		src[threadIdx.x] = src[threadIdx.x];
		min[threadIdx.x] = min[threadIdx.x];

	
	}
}

void pathfinder_host(){


	for (int t = 0; t < rows - 1; t++){
		pathfinder_iteration << <ceil((float)(cols) / BLOCKSIZE), BLOCKSIZE >> >(d_data + (t*cols), d_result, rows, cols);

		cudaDeviceSynchronize();
	}

	return;
}




void
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}



int main(int argc, char** argv)
{
	run(argc, argv);

	return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
	init(argc, argv);
	initCuda();

	unsigned long long cycles;
	double elapsed_time_host = -1;
	double elapsed_time_device = -1;

	int *src, *dst, *temp;
	int min;

	dst = h_result;
	src = new int[cols];

	startTime();
	for (int t = 0; t < rows - 1; t++) {
		temp = src;
		src = dst;
		dst = temp;
		for (int n = 0; n < cols; n++){
			min = src[n];
			if (n > 0)
				min = MIN(min, src[n - 1]);
			if (n < cols - 1)
				min = MIN(min, src[n + 1]);
			dst[n] = h_wall[t + 1][n] + min;
		}

		
	}

	stopTime(elapsed_time_host);


	startTime();
	pathfinder_host();
	cudaDeviceSynchronize();
	stopTime(elapsed_time_device);

	cudaMemcpy(h_result_cuda, d_result, sizeof(int)*cols, cudaMemcpyDeviceToHost);
	
	printf("host: %d %d %.2f\n", rows, cols, elapsed_time_host * 1e-3);
	printf("device: %d %d %.2f\n", rows, cols, elapsed_time_device * 1e-3);

	int test = TRUE;
	for (int i = 0; i < cols; i++){
		if (dst[i] != h_result_cuda[i]){
			test = FALSE;
		}
	}

	if (test)
		printf("TEST PASSED\n");
	else
		printf("TEST FAILED\n");

#ifdef BENCH_PRINT

	for (int i = 0; i < cols; i++)

		printf("%d ", data[i]);

	printf("\n");

	for (int i = 0; i < cols; i++)

		printf("%d ", dst[i]);

	printf("\n");

#endif

	cudaFree(d_data);
	cudaFree(d_result);

	delete[] h_result_cuda;
	delete[] h_data;
	delete[] h_wall;
	delete[] dst;
	delete[] src;

	system("pause");
}
