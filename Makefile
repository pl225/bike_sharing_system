CFLAGS = -O3
CC = nvcc
SRC = bss.c grafo.c solucao.c vizinhanca.c cuda.cu
OBJ = $(SRC:.c* = .o)

bss: $(OBJ)
	$(CC) -x cu $(CFLAGS) -o bss $(OBJ) -lm
clean:
	rm -f core *.o