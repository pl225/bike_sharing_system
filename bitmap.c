#include "bitmap.h"
#include <stdlib.h>
#include <stdio.h>

void set (Conjunto* c, int i) {
	int j = WORD_OFFSET(i);
	if (j >= c->n) {
		c->n = j+1;
		c->mapa = realloc(c->mapa, c->n * sizeof(naco));
	}
	c->mapa[j] |= (1 << BIT_OFFSET(i));
}

void clear (Conjunto* c, int i) {
	int j = WORD_OFFSET(i);
	if (j >= c->n) return;
	c->mapa[j] &= ~(1 << BIT_OFFSET(i));
}

int get (Conjunto* c, int i) {
	int j = WORD_OFFSET(i);
	if (j >= c->n) return 0;
	naco bits = c->mapa[j] & (1 << BIT_OFFSET(i));
	return bits != 0;
}

Conjunto* criarConjunto (int amplitude) {
	Conjunto* c = malloc(sizeof(Conjunto));
	int n = WORD_OFFSET(amplitude) + 1;
	c->n = n;
	c->mapa = malloc(sizeof(naco) * n);
	return c;
}

void liberarConjunto (Conjunto* c) {
	free(c->mapa);
	free(c);
}