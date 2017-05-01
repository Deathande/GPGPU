CC=gcc
LFLAGS=-lOpenCL
CFLAGS=-g

all: mc

mc: main.o
	$(CC) $+ -o $@ $(LFLAGS)

main.o: main.c
	$(CC) -c $(CFLAGS) $<
