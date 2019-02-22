#include "tabu.h"
#include <stdlib.h>
#include <stdio.h>

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

void liberarListaTabu (ListaTabu lista) {
	for (int i = 0; i < lista.tamanho; i++)
		if (lista.movimentos[i].n > 0)
			free(lista.movimentos[i].feitos);
	free(lista.movimentos);
	for (int i = 0; i < lista.nVertices * lista.nVertices; i++) 
		if (lista.arcos[i] != NULL)
			liberarConjunto(lista.arcos[i]);
	free(lista.arcos);
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

void atualizarListaTabu (ListaTabu* lista, int* caminho, int tamanhoCaminho) {
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
	
	if (a > move->n) move->feitos = realloc(move->feitos, sizeof(ArcoPosicao) * a);
	else if (a == 0) free(move->feitos);
	
	move->n = a;
	lista->indiceAtual = lista->indiceAtual + 1 >= lista->tamanho ? 0 : lista->indiceAtual + 1;
}