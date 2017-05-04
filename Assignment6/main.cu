#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

__device__ float function(float x)
{
  return x * x;
}

__global__ void init(unsigned int seed, curandState_t* states)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, blockIdx.x, 0, &states[id]);
}

__global__ void mc(int* ret,
                     curandState_t* states,
                     float a,
                     float b,
                     float h,
                     float (*f)(float))
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = curand_uniform(&states[i]) * (b-a) + a;
  float y = curand_uniform(&states[i]) * (h-0) + 0;
  if (function(x) >= y)
    atomicAdd(ret, 1);
}

int main()
{
  int* h_a;
  int* d_a;
  float a = 1.0;
  float b = 10.0;
  float h = 150.0;
  unsigned int grid = 1000;
  cudaError_t err;
  
  curandState_t* states;
  unsigned int n = BLOCK_SIZE * grid;

  h_a = (int*)malloc(sizeof(int));
  *h_a = 0;

  cudaMalloc((void**)&d_a, sizeof(int));
  cudaMalloc((void**)&states, n * sizeof(curandState_t));

  cudaMemcpy(d_a, h_a, sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(grid, 1, 1);

  init<<<dimGrid, dimBlock>>>((unsigned int) time(NULL), states);

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "init : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  mc<<<dimGrid, dimBlock>>>(d_a, states, a, b, h, &function);
  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "mc : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  cudaMemcpy(h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

  printf("return: %d\n", *h_a);
  float result = ((b - a) * h) * ((float)*h_a / (float)n);
  printf("answer: %f\n", result);
  cudaFree(d_a);
}
