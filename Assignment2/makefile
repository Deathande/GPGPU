CC=nvcc
CFLAGS=-g -c
MAT_SIZE=10

all: mat_mul read create

mat_mul: mat_mul.o g_matrix_mult.o
	$(CC) $+ -o $@

read: read_file.o
	$(CC) $+ -o $@

create: create_file.o
	$(CC) $+ -o $@

mat_mul.o: mat_mul.cu
	$(CC) $(CFLAGS) $<

read_file.o: read_file.c
	$(CC) $(CFLAGS) $<

create_file.o: create_file.c
	$(CC) $(CFLAGS) $<

g_matrix_mult.o: g_matrix_mult.cu g_matrix_mult.h
	$(CC) $(CFLAGS) $<

mat1.bin: create
	./create $@ $(MAT_SIZE)

mat2.bin: create
	./create $@ $(MAT_SIZE)

data: mat1.bin mat2.bin

run: mat_mul read mat1.bin mat2.bin
	./mat_mul $(MAT_SIZE) mat1.bin mat2.bin result.bin
	./read result.bin $(MAT_SIZE)

clean:
	rm -f *.o create mat_mul read *.bin
