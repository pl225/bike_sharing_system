#include "bitmap.h"
#include <stdlib.h>

typedef struct
{
	int i, j, k; // a(i, j) e posicao k
} ArcoPosicao;

typedef struct 
{
	ArcoPosicao* feitos;
	int n;
} Movimento;

typedef struct
{
	Conjunto* arcos;
	Movimento* movimentos;
	int tamanho;
	int nVertices;
	int indiceAtual;
} ListaTabu;

int indConj(int i, int j, int n) {
	return i * n + j;
}

ListaTabu criarListaTabu (int n, int nVertices) {
	ListaTabu lista;
	lista.tamanho = n;
	lista.nVertices = nVertices;
	lista.indiceAtual = 0;
	lista.movimentos = malloc(sizeof(Movimento) * n);
	for (int i = 0; i < n; i++) lista.movimentos[i].n = 0;
	lista.arcos = malloc(sizeof(Conjunto) * nVertices * nVertices);
	return lista;
}

void AtualizarListaTabu (ListaTabu* lista, int* caminho, int tamanhoCaminho) {
	Movimento* move = &lista->movimentos[lista->indiceAtual];
	if (move->n > 0) {
		for (int a = 0; a < move->n; a++) {
			clear(lista->arcos[indConj(move->feitos[a].i, move->feitos[a].j, lista->nVertices)], move->feitos[a].k);
		}
		free(move->feitos);
	}

	move->feitos = malloc(sizeof(ArcoPosicao) * tamanhoCaminho);
	Conjunto c;
	int a = 0;
	for (int i = 1; i < tamanhoCaminho; i++) {
		c = lista->arcos[indConj(caminho[i-1], caminho[i], lista->nVertices)];
		if (!get(c, i-1)) {
			set(c, i-1);
			move->feitos[a] = {caminho[i-1], caminho[i], i-1};
			a++;
		}
	}
	move->feitos = realloc(move->feitos, sizeof(Movimento) * a);
	lista->indiceAtual += 1;
}

int main(int argc, char const *argv[])
{
	/* code */
	return 0;
}