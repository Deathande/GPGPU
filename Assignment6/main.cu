#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

__global__
void mc_alg(float a, float b, float height, float (*f)(float), unsigned int* num)
{
  int c_i = threadIdx.x;
  curandState* state;
  float r = curand_uniform(state);
  *num = 5;
  printf("%d\n", *num);
  printf("here");
}

float function(float x)
{
  return x * x;
}

int main()
{
  float h_a;
  float h_b;
  float h_h;
  unsigned int* num;
  unsigned int h_n = 0;

  cudaMalloc((void**) &num, sizeof(unsigned int));
  cudaMemcpy(&h_n, num, sizeof(unsigned int), cudaMemcpyHostToDevice);

  mc_alg<<<1, 1024>>>(h_a, h_b, h_h, &function, num);
  cudaMemcpy(num, &h_n, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  printf("%d\n", h_n);
}
