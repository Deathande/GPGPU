#include "kernals.h"

// Maximum thread size on my GPU is 1024
// Change based on hardware
#define BLOCK_SIZE 1024

int* histogram(float* arr,
               float* mins,
               int size,
               int num_bins)
{
  float *d_arr;
  float *d_mins;
  int* result;
  int* d_result;
  int d_size;

  result = (int*)malloc(num_bins * sizeof(int));

  cudaMalloc((void**) &d_arr, size * sizeof(float));
  cudaMalloc((void**) &d_mins, (num_bins+1) * sizeof(float));
  cudaMalloc((void**) &d_result, num_bins * sizeof(int));

  // Ensure the result array is initialized to 0
  cudaMemset(d_result, 0, num_bins * sizeof(float));

  cudaMemcpy(d_arr, arr, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mins, mins, (num_bins+1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(ceil((float)num_bins / (float)dimBlock.x), 1, 1);
  printf("%d\n", dimGrid.x);

  g_histogram<<<dimGrid, dimBlock>>>(d_arr, d_mins, d_result, size);
  cudaMemcpy(result, d_result, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_arr);
  cudaFree(d_mins);
  cudaFree(d_result);
  return result;
}

float max(float* arr, unsigned int size)
{
  float result = arr[0];
  for (int i = 1; i < size; i++)
  {
    if (result < arr[i])
      result = arr[i];
  }
  return result;
}

float min(float* arr, unsigned int size)
{
  float result = arr[0];
  for (int i = 1; i < size; i++)
  {
    if (result > arr[i])
      result = arr[i];
  }
  return result;
}

__global__
void g_histogram(float* arr,
                 float* mins,
                 int* result,
                 int size)
{
  __shared__ float* s_arr;
  __shared__ float* s_mins;
  int id_x = blockIdx.x * blockDim.x + threadIdx.x;
  s_arr = arr;
  s_mins = mins;
  for (int i = 0; i < size; i++)
  {
    if (s_arr[i] >= s_mins[id_x] && s_arr[i] < s_mins[id_x+1])
        result[id_x]++;
  }
}
