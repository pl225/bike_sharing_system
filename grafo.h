#include <stdio.h>
#include <stdlib.h>

#define N_COLUNAS 2

typedef struct Grafo
{
	int n;
	int q;
	int *pontos;
	int *demandas;
} Grafo;

void liberarGrafo (Grafo g) { // mover
	free(g.pontos);
	free(g.demandas);
}

int IndicePontos (int i, int j) { // mover
	return i * N_COLUNAS + j;
}

Grafo carregarInstancia (char caminhoArquivo []) {

	FILE *arquivo = fopen(caminhoArquivo, "r");

	int n, q;

	fscanf(arquivo, "%d\n%d\n", &n, &q);

	Grafo g;
	g.n = n;
	g.q = q;//5;

	g.pontos = (int *) malloc(sizeof(int) * n * N_COLUNAS);
	g.demandas = (int *) malloc(sizeof(int) * n);
	float p1, p2;

	for (int i = 0; i < n; i++) {
		fscanf(arquivo, "%*d %f %f\n", &p1, &p2);
		g.pontos[IndicePontos(i, 0)] = (int) p1;
		g.pontos[IndicePontos(i, 1)] = (int) p2;
	}

	for (int i = 0; i < n; i++) {
		fscanf(arquivo, "%*d %d\n", &g.demandas[i]);
	}

	fclose(arquivo);

	return g;
}