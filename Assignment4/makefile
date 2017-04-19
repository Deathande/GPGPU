# Compiled on Arch Linux with nvcc version 8.0.44
CC=nvcc
CFLAGS=-g -c

all: read create hist

read: read_file.o
	$(CC) $+ -o $@

create: create_file.o
	$(CC) $+ -o $@

hist: histogram.o kernals.o
	$(CC) $+ -o $@

histogram.o: histogram.cu 
	$(CC) $(CFLAGS) $<

read_file.o: read_file.c
	$(CC) $(CFLAGS) $<

create_file.o: create_file.c
	$(CC) $(CFLAGS) $<

kernals.o: kernals.cu kernals.h
	$(CC) $(CFLAGS) $<

test: hist create
	./create test.dat 3000 && ./hist test.dat 3000 100

clean:
	rm -f *.o create hist read *.bin
