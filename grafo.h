#include <stdio.h>
#include <stdlib.h>

#define N_COLUNAS 2

typedef struct Grafo
{
	int n;
	int q;
	float *custos;
	int *demandas;
} Grafo;

void liberarGrafo (Grafo g) { // mover
	free(g.custos);
	free(g.demandas);
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