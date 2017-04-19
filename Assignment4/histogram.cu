#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernals.h"

void err(const char*);
float* read_dat(const char*, unsigned int);

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    printf("Usage: %s FILE NUM_ELEMENTS NUM_BINS\n", argv[0]);
    exit(0);
  }

  int size;
  int num_bins;
  float* arr;
  float* min_int;
  int* hist;
  float min_val;
  float max_val;
  float int_size;
  
  size = (int)strtol(argv[2], (char**)NULL, 10);
  if (size < 0)
    err("Invalid number of elements");
  num_bins = (int)strtol(argv[3], (char**)NULL, 10);
  if (num_bins < 0)
    err("Invalid number of bins");

  arr = read_dat(argv[1], size);

  // minimum of intervals plus the maximum value.
  // Essentially serves as our intervals because each
  // element will follow min_int[i] <= element < min_int[i+1]
  // By increading the actual size by one we don't have to worry
  // about indexing issues in the kernal when we do min_int[i+1]
  min_int = (float*)malloc(num_bins+1 * sizeof(float));
  min_val = min(arr, size);
  printf("min: %f\n", min_val);
  max_val = max(arr, size);
  printf("max: %f\n", max_val);
  int_size = (max_val - min_val) / num_bins;

  min_int[0] = min_val;
  for (int i = 1; i < num_bins; i++)
    min_int[i] = min_int[i-1] + int_size;
  min_int[num_bins] = max_val;

  clock_t t1 = clock();
  hist = histogram(arr, min_int, size, num_bins);
  clock_t t2 = clock();

  for (int i = 0; i < num_bins; i++)
    printf("%f <= x < %f: %d\n", min_int[i], min_int[i+1], hist[i]);
  printf("in time: %f", (double)(t2 - t1) / CLOCKS_PER_SEC);
}

float* read_dat(const char* file, unsigned int size)
{
  FILE* fd;
  float* arr;

  fd = fopen(file, "r");
  if (fd < 0)
    err("Could not open file");
  arr = (float*)malloc(size * sizeof(float));
  if(fread(arr, sizeof(float), size, fd) < size)
    err("Error reading file");

  return arr;
}

void err(const char* msg)
{
  perror(msg);
  exit(1);
}

