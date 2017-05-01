#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

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

float function(float x)
{
  return x * x;
}

int main()
{
  float h_a = 0;
  float h_b = 10;
  float h_h = 10;
  //float d_a;
  //float d_b;
  //float d_h;
  float ans;
  unsigned int* num;
  unsigned int h_n = 0;

  cudaMalloc((void**) &num, sizeof(unsigned int));
  cudaMemcpy(&h_n, num, sizeof(unsigned int), cudaMemcpyHostToDevice);

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(ceil((float)10 / (float)dimBlock.x), 1, 1);

  mc_alg<<<dimGrid,dimBlock>>>(h_a, h_b, h_h, &function, num);

  // can't get this to return anything...
  cudaMemcpy(num, &h_n, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  ans = h_n / (dimGrid.x * BLOCK_SIZE);
  printf("%f\n", ans);
  cudaFree(num);
}
