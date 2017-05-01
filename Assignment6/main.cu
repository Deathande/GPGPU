#include <cuda.h>
//#include <curand_kernel.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

/*
__global__
void mc_alg(float a, float b, float height, float (*f)(float), unsigned int* num)
{
  curandState* state;
  float x = (float)curand(state) / (float)RAND_MAX / (b-a) + a;
  float y = (float)curand(state) / (float)RAND_MAX / (height-0) + 0;
  printf("here");
  if (f(x) >= y)
    atomicAdd(num, 1);
}
*/

__global__ void test(int* arr)
{
  int i = threadIdx.x;
  printf("%d\n", i);
  arr[i] = i;
}

float function(float x)
{
  return x * x;
}

int main()
{
  int* h_a;
  int* d_a;
  unsigned int n = 5;

  h_a = (int*)malloc(sizeof(int) * n);
  cudaMalloc((void**)&d_a, sizeof(int) * n);
  test<<<1, n>>>(d_a);
  cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyHostToDevice);
  for (int i = 0; i < n; i++)
    printf("%d\n", h_a[i]);
  cudaFree(d_a);
}
