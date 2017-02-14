#include <cuda.h>
#include <stdio.h>

__global__
void g_scalar_mult(float* a, float* b)
{
  a[threadIdx.x] *= *b;
}

float* scalar_mult(const float scaler,
                   const float* vect,
		   unsigned int size)
{
  float* cuda_vect;
  float* cuda_scal;
  float* answer;

  answer = (float*)malloc(size * sizeof(float));
  
  cudaMalloc((void**)&cuda_vect, 4*sizeof(float));
  cudaMalloc((void**)&cuda_scal, 4*sizeof(float));

  cudaMemcpy(cuda_vect,
             vect,
	     size * sizeof(float),
             cudaMemcpyHostToDevice);

  cudaMemcpy(cuda_scal,
             &scaler,
	     size * sizeof(float),
             cudaMemcpyHostToDevice);

  g_scalar_mult<<<1, size>>>(cuda_vect, cuda_scal);
  // side effect?
  cudaMemcpy(answer,
             cuda_vect,
	     size * sizeof(float),
	     cudaMemcpyDeviceToHost);

  cudaFree(cuda_vect);
  cudaFree(cuda_scal);
  return answer;
}

int main()
{
  float* answer;
  float a[] = {1,2,3,4};
  float scal = 10;
  answer = scalar_mult(scal, a, 4);
  for (int i = 0; i < 4; i++)
    printf("%f\n", answer[i]);
  printf("\n");
}
