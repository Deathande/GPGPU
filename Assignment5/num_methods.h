#ifndef _NUM_METHODS
#define _NUM_METHODS

#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 1024
  
float trap_integration(float a, float b, unsigned int n, float(*f)(float));
float g_trap_integration(float a, float b, unsigned int n, float(*f)(float));
float g_sum(float* in, unsigned int size);
__global__ void g_sum_kernal(float* in, float* out, unsigned int size);
__global__ void trap_int_kernal(float a, float b, float dx, unsigned int n, float(*f)(float), float* out);

#endif
