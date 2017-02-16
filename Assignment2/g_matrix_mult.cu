#include "g_matrix_mult.h"

__global__ void g_mat_mult(double* m1, double* m2, double* m3, unsigned int size)
{
  int c_i = threadIdx.x;
  int c_j = threadIdx.y;
  double dot = 0;

  for (int i = 0; i < size; i++)
    dot += m1[c_i * size + i] * m2[i * size + c_j];
  m3[c_i*size + c_j] = dot;
}

__global__ void sg_mat_mult(double* m1,
                            double* m2,
                            double* m3,
                            unsigned int size)
{
  int c_i = threadIdx.x;
  int c_j = threadIdx.y;
  double dot = 0;

  for (int i = 0; i < size; i++)
    dot += m1[c_i * size + i] * m2[i * size + c_j];
  m3[c_i * size + c_j] = dot;
}

double* shared_matrix_mult(const double* m1,
                           const double* m2,
                           unsigned int size)
{
  int blocks = ceil(size * size / 1024);
  double* c_m1;
  double* c_m2;
  double* c_m3;
  double* result;
  unsigned int num_bytes;

  num_bytes = size * size * sizeof(double);
  result = (double*)malloc(num_bytes);
  cudaMalloc((void**) &c_m1, num_bytes);
  cudaMalloc((void**) &c_m2, num_bytes);
  cudaMalloc((void**) &c_m3, num_bytes);

  cudaMemcpy(c_m1, m1, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(c_m2, m2, num_bytes, cudaMemcpyHostToDevice);

  sg_mat_mult<<<blocks, size*size>>>(c_m1, c_m2, c_m3, size);

  cudaMemcpy(result, c_m3, num_bytes, cudaMemcpyDeviceToHost);

  cudaFree(c_m1);
  cudaFree(c_m2);
  cudaFree(c_m3);

  return result;
}

double* global_matrix_mult(const double* m1, const double* m2, unsigned int size)
{
  int blocks = ceil(size * size / 1024);
  double* c_m1;
  double* c_m2;
  double* c_m3;
  double* result;
  unsigned int num_bytes;

  num_bytes = size * size * sizeof(double);
  result = (double*)malloc(num_bytes);
  cudaMalloc((void**) &c_m1, num_bytes);
  cudaMalloc((void**) &c_m2, num_bytes);
  cudaMalloc((void**) &c_m3, num_bytes);

  cudaMemcpy(c_m1, m1, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(c_m2, m2, num_bytes, cudaMemcpyHostToDevice);

  g_mat_mult<<<blocks, size*size>>>(c_m1, c_m2, c_m3, size);
  cudaMemcpy(result, c_m3, num_bytes, cudaMemcpyDeviceToHost);

  cudaFree(c_m1);
  cudaFree(c_m2);
  cudaFree(c_m3);

  return result;
}

