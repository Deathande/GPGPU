#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main()
{
  srand(time(NULL));
  float random;
  float a = 4;
  float b = 10;
  while(1)
  {
    random = a + (float)rand() / ((float)RAND_MAX / (b - a));
    printf("%f\n", random);
    if (random > 9)
      exit(1);
  }
}
