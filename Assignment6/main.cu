#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void init(unsigned int seed, curandState_t* states)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, blockIdx.x, 0, &states[id]);
}

__global__ void test(int* ret,
                     curandState_t* states,
                     float a,
                     float b,
                     float h,
                     float (*f)(float))
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = curand_uniform(&states[i]) * (b-a) + a;
  float y = curand_uniform(&states[i]) * (h-0) + 0;
  //printf("ID: %d f(x): %f y: %f\n", i, f(x), y);

  printf("ID: %d x: %f y: %f\n", i, x, y);
  if (x * x < y)
  {
    atomicAdd(ret, 1);
    printf("%d\n", *ret);
  }
}

__device__ float function(float x)
{
  return x * x;
}

int main()
{
  int* h_a;
  int* d_a;
  float a = 1.0;
  float b = 10.0;
  float h = 13.0;
  
  curandState_t* states;
  unsigned int n = BLOCK_SIZE;

  h_a = (int*)malloc(sizeof(int));
  *h_a = 0;
  cudaMalloc((void**)&d_a, sizeof(int));
  cudaMalloc((void**)&states, n * sizeof(curandState_t));
  cudaMemcpy(d_a, h_a, sizeof(int), cudaMemcpyHostToDevice);
  init<<<BLOCK_SIZE, 1>>>((unsigned int) time(NULL), states);
  test<<<BLOCK_SIZE, 1>>>(d_a, states, a, b, h, &function);
  cudaMemcpy(h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
  printf("return: %d\n", *h_a);
  float result = (b - a) * h * ((float)*h_a / (float)n);
  printf("answer: %f\n", result);
  cudaFree(d_a);
}
