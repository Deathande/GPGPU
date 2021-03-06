// Borrowed and slightly modified your file IO code from your website.
// g_matrix_mult.h and g_matrix_mult.cu have my matrix multiplication
// kernels. 
//
// Compiled and run on Arch Linux with nvcc version 8.0.44. It does not 
// run on the GPU cluster because the cluster's nvcc version does
// not allow c99 conventions.

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include "g_matrix_mult.h"

void fatal_error (const char* message);
// precondition: message is not NULL
// postcondition: message has been written to standard error & program terminated

char* create_error_string (const char* message, const char* data_string);

double* get_matrix_from_file (const char* file_name, int size);

double* multiply (double* m1, double* m2, int size);

void write_product_to_file (const char* file_name, double* product, int size);

int main (int argc, char** argv)
{
  if (argc != 5)
    fatal_error("assn2 <matrix_size> <matrix_file_1> <matrix_file_2> <result_file\n");
  int size = (int) strtol(argv[1], (char **)NULL, 10);
  if (size <= 0)
    fatal_error ("invalid matrix size");
  double* m1 = get_matrix_from_file (argv[2], size);
  if (m1 == NULL)
    fatal_error (create_error_string ("cannot get matrix from file %s", argv[2]));
  double* m2 = get_matrix_from_file (argv[3], size);
  if (m2 == NULL)
    fatal_error (create_error_string ("cannot get matrix from file %s", argv[3]));

  time_t t1, t2;

  // Serial matrix multiplication
  t1 = clock();
  double* result = multiply (m1, m2, size);
  t2 = clock();
  printf("serial time: %f\n", (float)(t2 - t1) / CLOCKS_PER_SEC);
  //

  // device matrix multiplication, global memory
  t1 = clock();
  double* result2 = global_matrix_mult(m1, m2, size);
  t2 = clock();
  printf("global time: %f\n", (float)(t2 - t1) / CLOCKS_PER_SEC);
  //

  // device matrix multiplication, shared memory
  t1 = clock();
  double* result3 = shared_matrix_mult(m1, m2, size);
  t2 = clock();
  printf("shared time: %f\n", (float)(t2 - t1) / CLOCKS_PER_SEC);
  //

  write_product_to_file (argv[4], result3, size);

  free (result);
  free (result2);
  free (result3);
  free (m2);
  free (m1);
  return 0;
}

void fatal_error (const char* message)
{
  fprintf (stderr, message);
  exit (0);
}

char* create_error_string (const char* message, const char* data_string)
{
  char* result = (char*)malloc (60 * sizeof(char));
  if (result == NULL)
    fatal_error ("malloc error");
  snprintf (result, 60, message, data_string);
  return result;
}

double* get_matrix_from_file (const char* file_name, int size)
{
  int fd = open (file_name, O_RDONLY);
  if (fd < 0)
    fatal_error (create_error_string ("could not open %s", file_name));
  double* m = (double*)malloc (size * size * sizeof (double));
  if (m == 0)
    fatal_error ("malloc error");
  size_t read_size = read(fd, m, size * size * sizeof(double));
  if (read_size < size * size * sizeof(double))
    fatal_error ("error reading from file");
  close (fd);
  return m;
}

double* multiply (double* m1, double* m2, int size)
{
  double* result;
  double dot;
  result = (double*)malloc(size * size * sizeof(double));
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      dot = 0;
      for (int k = 0; k < size; k++)
        dot += m1[i*size + k] * m2[k*size + j];
      result[i*size + j] = dot;
    }
  }
  return result;
}

void write_product_to_file (const char* file_name, double* product, int size)
{
  int fd = creat (file_name, 0666);
  if (fd < 0)
    fatal_error (create_error_string ("could not open %s", file_name));
  size_t write_size = write (fd, product, size * size * sizeof(double));
  if (write_size < size * size * sizeof(double))
    fatal_error ("error writing to file");
  close (fd);
}
