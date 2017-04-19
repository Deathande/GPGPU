#include "num_methods.h"

float trap_integration(float a, float b, unsigned int n, float (*f)(float))
{
  float dx = (b - a) / n;
  float sum = f(a) + f(b);
  for (int i = 1; i < n-1; i++)
    sum += 2 * f(a + i * dx);
  sum *= 0.5 * dx;
  return sum;
}

float g_sum(float* in, unsigned int size)
{
  float* c_in;
  float* c_out;
  float* result;

  unsigned int num_bytes = sizeof(float) * size;
  cudaMalloc((void**) &c_in, num_bytes);
  cudaMalloc((void**) &c_out, num_bytes);
  cudaMemcpy(c_in, in, num_bytes, cudaMemcpyHostToDevice);
  g_sum_kernal<<<BLOCK_SIZE, BLOCK_SIZE>>> (c_in, c_out, size);
  cudaDeviceSynchronize();
  cudaMemcpy(result, c_out, num_bytes, cudaMemcpyDeviceToHost);
  cudaFree(c_in);
  cudaFree(c_out);
  return result[0];
}

float g_trap_integration(float a, float b, unsigned int n, float(*f)(float))
{
  /*
  float* result;
  float* out;
  float dx = (b - a) / n;
  cudaMalloc((void**) &out, sizeof(float) * n);
  trap_int_kernal<<<BLOCK_SIZE, BLOCK_SIZE>>>(a, b, dx, n, f, out);
  cudaMemcpy(result, out, sizeof(float) * n, cudaMemcpyDeviceToHost);
  cudaFree(out);
  cudaFree(result);
  return 0.5 * dx * result[0];
  */
  return 0;
}

__global__
void g_sum_kernal(float* in, float* out, unsigned int size)
{
    __shared__ float in_s[2*BLOCK_SIZE];
    int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    in_s[threadIdx.x]            = ((idx              < size)? in[idx]:            0.0f);
    in_s[threadIdx.x+BLOCK_SIZE] = ((idx + BLOCK_SIZE < size)? in[idx+BLOCK_SIZE]: 0.0f);

    for(int stride = 1; stride < BLOCK_SIZE<<1; stride <<= 1) {
      __syncthreads();
    if(threadIdx.x % stride == 0)
      in_s[2*threadIdx.x] += in_s[2*threadIdx.x + stride];
    }
    if (threadIdx.x == 0)
      out[blockIdx.x] = in_s[0];
}

__global__
void trap_int_kernal(float a, float b, float dx, unsigned int n, float(*f)(float), float* out)
{
}
