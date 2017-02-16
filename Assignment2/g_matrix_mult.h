#ifndef __G_MATRIX_MULTIPLICATION
#define __G_MATRIX_MULTIPLICATION

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

// Perform matrix multiplication using global device memory
double* global_matrix_mult(const double*, const double*, unsigned int);
double* shared_matrix_mult(const double*, const double*, unsigned int);

__global__ void g_mat_mult(double* m1, 
                           double* m2,
			   double* m3,
			   unsigned int size);
__global__ void sg_mat_mult(double*, double*, double*, unsigned int);

#endif
