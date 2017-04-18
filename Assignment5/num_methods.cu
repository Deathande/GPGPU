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
