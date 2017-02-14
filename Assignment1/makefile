# Compiled on Arch Linux, nvcc version 8.0.44

CC=nvcc
CFLAGS=-g -c

all: mm

mm: main.o
	$(CC) main.o -o $@

main.o: main.cu
	$(CC) $(CFLAGS) main.cu

clean:
	rm -f *.o
