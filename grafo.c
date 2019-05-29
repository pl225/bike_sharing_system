#include <stdio.h>
#include <stdlib.h>
#include "grafo.h"

void liberarGrafo (Grafo g) {
	free(g.custos);
	free(g.demandas);
}

int IndicePontos (int i, int j) {
	return i * N_COLUNAS + j;
}

Grafo carregarInstancia(char caminhoArquivo []) {

	FILE *arquivo = fopen(caminhoArquivo, "r");
	int n, q, d;

	fscanf(arquivo, "%d\n", &n);

	Grafo g;
	g.n = n;

	g.demandas = (int *) malloc(sizeof(int) * n);
	int demanda0 = 0;

	for (int i = 0; i < n; i++) {
		fscanf(arquivo, "%d", &d);
		g.demandas[i] = -1 * d;
		demanda0 += g.demandas[i];
		fgetc(arquivo);
	}
	g.demandas[0] = -1 * demanda0;

	fscanf(arquivo, "%d\n", &q);
	g.q = q;

	int a = 0;
	g.custos = malloc(sizeof(float) * n * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++, a++) {
			fscanf(arquivo, "%f", &g.custos[a]);
			fgetc(arquivo);
			if (i == j) g.custos[a] = 0;
		}
	}

	fclose(arquivo);

	return g;
}