#ifndef __G_MATRIX_MULTIPLICATION
#define __G_MATRIX_MULTIPLICATION

#include <stdlib.h>
#include <cuda.h>

double* global_matrix_mult(const double*, const double*, unsigned int);
__global__ void g_mat_mult(double* m1, double* m2, double* m3, unsigned int size);

#endif
