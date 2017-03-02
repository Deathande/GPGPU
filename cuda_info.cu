#include <stdio.h>
#include <cuda.h>

void print_caps(const cudaDeviceProp* props);

int main()
{
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  //printf("%d\n", props.maxThreadsPerBlock);
  print_caps(&props);
}

void print_caps(const cudaDeviceProp* props)
{
  printf("name: %s\n", props->name);
  printf("shared memory per block: %.2fKB\n", (float)props->sharedMemPerBlock / 1024);
  printf("total global memory: %.2fMB\n", (float)props->totalGlobalMem / 1048576);
  printf("regs per block: %d\n", props->regsPerBlock);
  printf("Warp size: %d\n", props->warpSize);
  printf("Max threads per block: %d\n", props->maxThreadsPerBlock);
  printf("max thread dimention: %dx%dx%d\n", props->maxThreadsDim[0], props->maxThreadsDim[1], props->maxThreadsDim[2]);
  printf("max grid size: %dx%dx%d\n", props->maxThreadsDim[0], props->maxThreadsDim[1], props->maxThreadsDim[2]);
  printf("clock rate: %dKHz\n", props->clockRate);
}
