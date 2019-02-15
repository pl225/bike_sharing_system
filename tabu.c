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
	Conjunto** arcos;
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
	lista.arcos = malloc(sizeof(Conjunto*) * nVertices * nVertices);
	return lista;
}

ArcoPosicao criarArcoPosicao (int i, int j, int k) {
	ArcoPosicao a;
	a.i = i, a.j = j, a.k = k;
	return a;
}

void preencherListaTabu (ListaTabu* lista, int* caminho, int tamanhoCaminho) {
	
	for (int i = 0; i < lista->nVertices * lista->nVertices; i++)
			lista->arcos[i] = criarConjunto(tamanhoCaminho);
	
	Movimento* move = &lista->movimentos[0];
	move->feitos = malloc(sizeof(ArcoPosicao) * (tamanhoCaminho - 1));
	move->n = tamanhoCaminho - 1;
	lista->indiceAtual = 1;
	
	int a = 0;
	for (int i = 1; i < tamanhoCaminho; i++) {
		a = i - 1;
		set(lista->arcos[indConj(caminho[a], caminho[i], lista->nVertices)], a);
		move->feitos[a] = criarArcoPosicao(caminho[a], caminho[i], a);
	}
}

void AtualizarListaTabu (ListaTabu* lista, int* caminho, int tamanhoCaminho) {
	Movimento* move = &lista->movimentos[lista->indiceAtual];
	if (move->n > 0) {
		for (int a = 0; a < move->n; a++) {
			clear(lista->arcos[indConj(move->feitos[a].i, move->feitos[a].j, lista->nVertices)], move->feitos[a].k);
		}
		free(move->feitos);
	}

	move->feitos = malloc(sizeof(ArcoPosicao) * (tamanhoCaminho - 1));
	Conjunto* c;
	int a = 0, b = 0;
	for (int i = 1; i < tamanhoCaminho; i++) {
		b = i - 1;
		c = lista->arcos[indConj(caminho[b], caminho[i], lista->nVertices)];
		if (!get(c, b)) {
			set(c, b);
			move->feitos[a] = criarArcoPosicao(caminho[b], caminho[i], b);
			a++;
		}
	}
	move->feitos = realloc(move->feitos, sizeof(ArcoPosicao) * a);
	move->n = a;
	lista->indiceAtual = lista->indiceAtual + 1 >= lista->tamanho ? 0 : lista->indiceAtual + 1;
}

int main(int argc, char const *argv[])
{
	
	return 0;
}