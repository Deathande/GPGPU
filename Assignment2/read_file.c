#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>


void fatal_error (const char* message);
// precondition: message is not NULL
// postcondition: message has been written to standard error & program terminated

void display_matrix (const double* m, const int size);

int main (int argc, char** argv)
{
  if (argc < 3)
    fatal_error ("read_file <file_name> <size>\n");
  int size = (int) strtol(argv[2], (char **)NULL, 10);
  if (size <= 0)
    fatal_error ("invalid matrix size");
  double* m = malloc (size * size * sizeof(double));
  if (m == NULL)
    fatal_error ("malloc error");
  int fd = open (argv[1], O_RDONLY);
  size_t read_size = read(fd, m, size * size * sizeof(double));
  if (read_size < size * size * sizeof(double))
    fatal_error ("error reading from file");
  display_matrix (m, size);
  close (fd);
  free (m);
  return 0;
}

void fatal_error (const char* message)
{
  fprintf (stderr, message);
  exit (0);
}

void display_matrix (const double* m, const int size)
{
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
      printf ("%lf ", m[i*size + j]);
    printf ("\n");
  }
}

