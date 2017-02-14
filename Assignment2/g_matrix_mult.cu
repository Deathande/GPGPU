#include "g_matrix_mult.h"

double* global_matrix_mult(const double* m1, const double* m2, unsigned int size)
{
  float* c_m1;
  float* c_m2;
  float* result;
  unsigned int num_bytes;

  num_bytes = size * size * sizeof(double);
  result = malloc(num_bytes);
  cudaMalloc((void**) &c_m1, num_bytes);
  cudaMalloc((void**) &c_m2, num_bytes);

  cudaMemcpy(c_m1, m1, num_bytes);
  cudaMemcpy(c_m2, m2, num_bytes);

  

  cudaFree(c_m1);
  cudaFree(c_m2);
}

