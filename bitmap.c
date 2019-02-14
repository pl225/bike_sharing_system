#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef uint32_t naco;
enum { BITS_PER_WORD = sizeof(naco) * CHAR_BIT};
#define WORD_OFFSET(b) ((b) / BITS_PER_WORD)
#define BIT_OFFSET(b) ((b) % BITS_PER_WORD)

typedef struct 
{
	naco* mapa;
	int n;
} Conjunto;

void set (Conjunto *c, int i) {
	int j = WORD_OFFSET(i);
	if (j >= c->n) {
		c->n = j+1;
		c->mapa = realloc(c->mapa, c->n * sizeof(naco));
	}
	c->mapa[j] |= (1 << BIT_OFFSET(i));
}

void clear (Conjunto c, int i) {
	int j = WORD_OFFSET(i);
	if (j >= c.n) return;
	c.mapa[j] &= ~(1 << BIT_OFFSET(i));
}

int get (Conjunto c, int i) {
	int j = WORD_OFFSET(i);
	if (j >= c.n) return 0;
	naco bits = c.mapa[j] & (1 << BIT_OFFSET(i));
	return bits != 0;
}

Conjunto criarConjunto (int amplitude) {
	Conjunto c;
	int n = WORD_OFFSET(amplitude) + 1;
	c.n = n;
	c.mapa = malloc(sizeof(naco) * n);
	return c;
}

void liberarConjunto (Conjunto c) {
	free(c.mapa);
}

int main(int argc, char const *argv[])
{
	Conjunto c = criarConjunto(30);
	set(&c, 2000);
	printf("%d\n", get(c, 2000));
	clear(c, 25);
	printf("%d\n", get(c, 25));

	liberarConjunto(c);
	return 0;
}
