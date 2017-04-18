CC=nvcc
CFLAGS=-g -w
LFLAGS=

all: nm

nm: main.o num_methods.o
	$(CC) $(LFLAGS) $+ -o $@

main.o: main.cu
	$(CC) $(CFLAGS) -c $<

num_methods.o: num_methods.cu num_methods.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o nm
