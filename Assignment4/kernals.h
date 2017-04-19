#include <cuda.h>
#include <math.h>
#include <stdio.h>

float max(float* arr, unsigned int);
float min(float* arr, unsigned int);

int* histogram(float* arr,
               float* mins,
               int size,
               int num_bins);

__global__
void g_histogram(float* arr,
                 float* mins,
                 int* result,
                 int size);
