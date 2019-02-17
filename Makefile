CFLAGS = -O3
CC = gcc
SRC = bss.c vizinhanca.c solucao.c grafo.c
OBJ = $(SRC:.c = .o)

bss: $(OBJ)
	$(CC) $(CFLAGS) -o bss $(OBJ) -lm
clean:
	rm -f core *.o