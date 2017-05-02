CC=nvcc
CFLAGS=-g
LFLAGS=

all: mc

mc: main.o
	$(CC) $+ -o $@ $(LFLAGS)

main.o: main.cu
	$(CC) $(CFLAGS) -c $< 

clean:
	rm -f *.o mc
