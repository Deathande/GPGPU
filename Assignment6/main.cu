#include <cuda.h>
#include <cudarand.h>
#include <stdio.h>

__global__
void mc_alg(float a, float b, float height, float (*f)(float), unsigned int* num)
{
  int c_i = threadIdx.x;
  cudarandState state;
}

int main()
{
}
