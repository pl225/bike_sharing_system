#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef struct Grafo
{
	int n;
	int q;
	float *pontos;
	int *demandas;
} Grafo;

Grafo carregarInstancia (char caminhoArquivo []) {

	FILE *arquivo = fopen(caminhoArquivo, "r");

	int n, q;

	fscanf(arquivo, "%d\n%d\n", &n, &q);
	printf("%d %d\n", n, q);

	float pontos [n][2];
	int demandas [n];

	for (int i = 0; i < n; i++) {
		fscanf(arquivo, "%*d %f %f\n", &pontos[i][0], &pontos[i][1]);
	}

	for (int i = 0; i < n; i++) {
		fscanf(arquivo, "%*d %d\n", &demandas[i]);
	}

	Grafo g;
	g.n = n;
	g.q = q;

	g.pontos = (float *) malloc(sizeof(float) * n * n);
	memcpy(g.pontos, pontos, sizeof(pontos));
	g.demandas = (int *) malloc(sizeof(int) * n);
	memcpy(g.demandas, demandas, sizeof(demandas));

	fclose(arquivo);

	return g;
}