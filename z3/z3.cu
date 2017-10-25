
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define TRUE 1
#define FALSE 0


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


int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

void Usage(int argc, char**argv){

	fprintf(stderr, "Usage: %s <input_file> \n", argv[0]);

}



////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	no_of_nodes = 0;
	edge_list_size = 0;

	BFSGraph(argc, argv);
}

void bfsSeq(
	struct Node* h_graph_nodes,
	bool *h_graph_mask,
	bool *h_updating_graph_mask,
	bool *h_graph_visited,
	int  *h_graph_edges,
	int  *h_cost
	)
{
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = false;

		for (int tid = 0; tid < no_of_nodes; tid++)
		{
			if (h_graph_mask[tid] == true){
				h_graph_mask[tid] = false;
				for (int i = h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++)
				{
					int id = h_graph_edges[i];
					if (!h_graph_visited[id])
					{
						h_cost[id] = h_cost[tid] + 1;
						h_updating_graph_mask[id] = true;
					}
				}
			}
		}

		for (int tid = 0; tid< no_of_nodes; tid++)
		{
			if (h_updating_graph_mask[tid] == true){
				h_graph_mask[tid] = true;
				h_graph_visited[tid] = true;
				stop = true;
				h_updating_graph_mask[tid] = false;
			}
		}

	} while (stop);
}

__global__ void bfsIteration(
	struct Node *d_graph_nodes,
	bool *d_graph_mask,
	bool *d_updating_graph_mask,
	bool *d_graph_visited,
	int  *d_graph_edges,
	int  *d_cost,
	int no_of_nodes
	)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < no_of_nodes){
		if (d_graph_mask[tid] == true){
			d_graph_mask[tid] = false;
			for (int i = d_graph_nodes[tid].starting; i<(d_graph_nodes[tid].no_of_edges + d_graph_nodes[tid].starting); i++)
			{
				int id = d_graph_edges[i];
				if (!d_graph_visited[id])
				{
					d_cost[id] = d_cost[tid] + 1;
					d_updating_graph_mask[id] = true;
				}
			}
		}
	}

}

#ifdef _SM_35_
	
__global__ void bfsCheck(
	bool *d_graph_mask,
	bool *d_updating_graph_mask,
	bool *d_graph_visited,
	int no_of_nodes
	)
{
	__shared__ extern bool stop;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < no_of_nodes){
		if (d_updating_graph_mask[tid] == true){
			d_graph_mask[tid] = true;
			d_graph_visited[tid] = true;
			stop = true;
			d_updating_graph_mask[tid] = false;
		}
	}
}



__global__ void bfsStart(
struct Node *d_graph_nodes,
	bool *d_graph_mask,
	bool *d_updating_graph_mask,
	bool *d_graph_visited,
	int  *d_graph_edges,
	int  *d_cost,
	int no_of_nodes
	)
{
	__shared__ bool stop;
	int numBlocks = ceil((float)(no_of_nodes) / 256.0);
	int numThreadsPerBlock = 256;
	do{
		stop = false;
		bfsIteration << <numBlocks, numThreadsPerBlock >> >(d_graph_nodes, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_graph_edges, d_cost, no_of_nodes);
		bfsCheck << <numBlocks, numThreadsPerBlock >> >(d_graph_mask, d_updating_graph_mask, d_graph_visited, no_of_nodes);
	} while (stop);
}

void bfsCuda(
	struct Node *d_graph_nodes,
	bool *d_graph_mask,
	bool *d_updating_graph_mask,
	bool *d_graph_visited,
	int  *d_graph_edges,
	int  *d_cost,
	int no_of_nodes
	)
{
	bfsStart<<<1, 1>>>(d_graph_nodes, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_graph_edges, d_cost, no_of_nodes);
}
#else

__global__ void bfsCheck(
	bool *d_graph_mask,
	bool *d_updating_graph_mask,
	bool *d_graph_visited,
	int no_of_nodes,
	bool *stop
	)
{
	*stop = false;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < no_of_nodes){
		if (d_updating_graph_mask[tid] == true){
			d_graph_mask[tid] = true;
			d_graph_visited[tid] = true;
			*stop = true;
			d_updating_graph_mask[tid] = false;
		}
	}
}


void bfsCuda(
struct Node *d_graph_nodes,
	bool *d_graph_mask,
	bool *d_updating_graph_mask,
	bool *d_graph_visited,
	int  *d_graph_edges,
	int  *d_cost,
	int no_of_nodes
	)
{
	int numBlocks = ceil((float)(no_of_nodes) / 256.0);
	int numThreadsPerBlock = 256;

	bool *d_stop;
	bool h_stop;
	cudaMalloc(&d_stop, sizeof(bool));
	do{
		bfsIteration <<<numBlocks, numThreadsPerBlock >>>(d_graph_nodes, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_graph_edges, d_cost, no_of_nodes);
		cudaDeviceSynchronize();
		bfsCheck <<<numBlocks, numThreadsPerBlock>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, no_of_nodes, d_stop);
		cudaDeviceSynchronize();
		cudaMemcpy(&h_stop, d_stop, sizeof(bool), cudaMemcpyDeviceToHost);
	} while (h_stop);

	cudaFree(d_stop);
}
#endif






////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(int argc, char** argv)
{
	char *input_f;
	int	 num_omp_threads;

	if (argc != 2){
		Usage(argc, argv);
		exit(0);
	}

	input_f = argv[1];

	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f, "r");
	if (!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp, "%d", &no_of_nodes);

	// allocate host memory
	struct Node* h_graph_nodes = (struct Node*) malloc(sizeof(struct Node)*no_of_nodes);
	bool *h_graph_mask = (bool*)malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*)malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*)malloc(sizeof(bool)*no_of_nodes);

	// allocate device memory
	struct Node* d_graph_nodes;
	bool *d_graph_mask;
	bool *d_updating_graph_mask;
	bool *d_graph_visited;

	cudaMalloc((void**)&d_graph_nodes, sizeof(struct Node)*no_of_nodes);
	cudaMalloc((void**)&d_graph_mask, sizeof(bool)*no_of_nodes);
	cudaMalloc((void**)&d_updating_graph_mask, sizeof(bool)*no_of_nodes);
	cudaMalloc((void**)&d_graph_visited, sizeof(bool)*no_of_nodes);

	

	int start, edgeno;
	// initalize the memory
	for (unsigned int i = 0; i < no_of_nodes; i++)
	{
		fscanf(fp, "%d %d", &start, &edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i] = false;
		h_updating_graph_mask[i] = false;
		h_graph_visited[i] = false;
	}

	//read the source node from the file
	fscanf(fp, "%d", &source);
	source = 0;

	//set the source node as true in the mask
	h_graph_mask[source] = true;
	h_graph_visited[source] = true;

	fscanf(fp, "%d", &edge_list_size);

	int id, cost;
	int* h_graph_edges = (int*)malloc(sizeof(int)*edge_list_size);
	for (int i = 0; i < edge_list_size; i++)
	{
		fscanf(fp, "%d", &id);
		fscanf(fp, "%d", &cost);
		h_graph_edges[i] = id;
	}

	if (fp)
		fclose(fp);

	

	// Initialize device memory
	int* d_graph_edges;
	cudaMalloc((void**)&d_graph_edges, sizeof(int)*edge_list_size);

	cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(struct Node)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);




	// allocate mem for the result on host side
	int* h_cost = (int*)malloc(sizeof(int)*no_of_nodes);
	for (int i = 0; i<no_of_nodes; i++)
		h_cost[i] = -1;
	h_cost[source] = 0;

	// allocate memory for the result on the device side
	int* h_cost_cuda = (int*)malloc(sizeof(int)*no_of_nodes);
	int* d_cost;

	cudaMalloc((void**)&d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy(d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice);

	double elapsed_time_host, elapsed_time_device;

	printf("Start sequential traversal of the tree\n");
	starttime();
	bfsSeq(h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_graph_edges, h_cost);
	stoptime(elapsed_time_host);
	printf("Start parallel traversal of the tree\n");
	starttime();
	bfsCuda(d_graph_nodes, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_graph_edges, d_cost, no_of_nodes);
	stoptime(elapsed_time_device);

	cudaMemcpy(h_cost_cuda, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);

	printf("Print results\n");
	//Store the result into a file
	FILE *h_fpo = fopen("h_result.txt", "w");
	FILE *d_fpo = fopen("d_result.txt", "w");
	int test = TRUE;
	for (int i = 0; i < no_of_nodes; i++){
		fprintf(h_fpo, "%d) cost:%d\n", i, h_cost[i]);
		fprintf(d_fpo, "%d) cost:%d\n", i, h_cost_cuda[i]);
		if (h_cost[i] != h_cost_cuda[i]){
			test = FALSE;
		}
	}
	fclose(h_fpo);
	fclose(d_fpo);
	printf("Result stored in result.txt\n");

	if (test){
		printf("TEST PASSED\n");
	}
	else {
		printf("TEST FAILED\n");
	}

	printf("Seq: %ld\tPar: %ld\n", elapsed_time_host, elapsed_time_device);

	// cleanup memory
	free(h_graph_nodes);
	free(h_graph_edges);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	free(h_cost);

	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);

	printf("Press any key...\n");
	getchar();

}