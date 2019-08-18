CFLAGS = -O3
CC = nvcc
SRC = grafo.c solucao.c vizinhanca.c cuda.c
OBJ = $(SRC:.c = .o)

bss: $(OBJ)
	$(CC) -x cu $(CFLAGS) -o bss bss.cu $(OBJ) -lm
clean:
	rm -f core *.o