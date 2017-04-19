#include <stdio.h>
#include <stdlib.h>
#include "num_methods.h"

float function(float x)
{
  return x * x; // x ** 2
}

int main(int argc, char** argv)
{
  float num_int;
  float a = 0;
  float b = 5;
  unsigned int n = 1000;
  num_int = trap_integration(a, b, n, &function);
  printf("a = %.2f b = %.2f n = %d\n", a, b, n);
  printf("f(x) = x * x\n");
  printf("Area under curve = %f\n", num_int);

  num_int = g_trap_integration(a, b, n, &function);
  printf("GPU Area under curve = %f\n", num_int);

  printf("---------------------------------\n");

  unsigned int size = 100;
  float arr[size];
  float sum = 0;
  for (int i = 0; i < size; i++)
    arr[i] = i;
  for (int i = 0; i < size; i++)
    sum += arr[i];
  printf("sum: %f\n", sum);
  float ans = g_sum(arr, size);
  printf("%f\n", ans);
}
