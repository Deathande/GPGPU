#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

void fatal_error (const char* message);
// precondition: message is not NULL
// postcondition: message has been written to standard error & program terminated

double* create_matrix (int size);

int main (int argc, char** argv)
{
  if (argc < 3)
    fatal_error ("create_file <file_name> <size>\n");
  int fd = creat (argv[1], 0666);
  if (fd < 0)
    fatal_error ("error creating file");
  int size = (int) strtol(argv[2], (char **)NULL, 10);
  if (size <= 0)
    fatal_error ("invalid matrix size");
  double* data = create_matrix (size);
  size_t write_size = write (fd, data, size * size * sizeof (double));
  if (write_size < size * size * sizeof(double))
    fatal_error ("error writing to file");
  free (data);
  close (fd);
  return 0;
}

void fatal_error (const char* message)
{
  fprintf (stderr, message);
  exit (0);
}

double* create_matrix (int size)
{
  double* m = malloc (size * size * sizeof(double));
  if (m == NULL)
    fatal_error ("malloc error");
  srand (time(NULL));
  for (int i = 0; i < size * size; ++i)
    m[i] = (float)rand() / RAND_MAX;
  return m;
}
