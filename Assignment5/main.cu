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
}
