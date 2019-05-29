#ifndef GRAFO_H
#define GRAFO_H
#include "grafo.h"
#endif

#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#ifndef VIZINHANCA_H
#define VIZINHANCA_H
#include "vizinhanca.h"
#endif

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h> 

void vns(Grafo g, FabricaSolucao fs, float results[])
{
	clock_t start = clock();

	Solucao x = GRASP(fs), xLinha, xLinhaLinha;

	const int kMax = 2;
	int k, maxIteracoes = 100 * g.n, i = 0;
	float melhorCusto = INFINITY;
	
	do {
		k = 1;

		do {
			xLinha = k == 0 ? _3OPT_P(x, fs) : splitP(x, fs);
			xLinhaLinha = RVND(xLinha, fs);

			if (xLinha.caminho != x.caminho) liberarSolucao(xLinha);
			
			if (xLinhaLinha.custo < x.custo && isViavel(xLinhaLinha)) {
				liberarSolucao(x);
				x = copiarSolucao(xLinhaLinha);
				if (xLinha.caminho != xLinhaLinha.caminho) liberarSolucao(xLinhaLinha);
				k = 1;
			} else {
				k += 1;
				if (xLinha.caminho != xLinhaLinha.caminho) liberarSolucao(xLinhaLinha);
			}
		} while (k < kMax);

		if (melhorCusto < x.custo) {
			i = 0;
			melhorCusto = x.custo;
		} else {
			i++;
		}

	} while (i < maxIteracoes);

	liberarSolucao(x);

	clock_t end = clock();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;
	
	results[0] = x.custo, results[1] = seconds;
}

int main(int argc, char *argv[])
{
	srand(time(NULL));
	float results[2]; // 0 = custo, 1 = tempo
	FILE *arq = fopen(argv[1], "w");

	char caminho[50];

	DIR *d;
  	struct dirent *dir;
  	d = opendir("InstancesBRP");
  	int i = 1;

	while ((dir = readdir(d)) != NULL) {
		if (dir->d_type == DT_REG) {
			strcpy(caminho, "InstancesBRP/");
			strcat(caminho, dir->d_name);	

			Grafo g = carregarInstancia(caminho);
			FabricaSolucao fs = instanciarFabrica(g);

			fprintf(arq, "Instancia: %s, Q: %d\n", caminho, g.q);
			float mediaTempo = 0;
			float melhorCusto = INFINITY;

			for (int l = 0; l < 10; l++) {
				vns(g, fs, results);
				fprintf(arq, "\tCusto: %.f, Tempo: %f\n", results[0], results[1]);
				if (results[0] < melhorCusto) melhorCusto = results[0];
				mediaTempo += results[1];
			}
			fprintf(arq, "Melhor custo: %.f, média dos tempos: %f\n", melhorCusto, mediaTempo / 10.0);
			printf("Instancia: %s, Q: %d terminada. %d/50\n", caminho, g.q, i);

			liberarGrafo(g);
			liberarFabrica(fs);
			strcpy(caminho, "");
			i++;
		}
	}
	printf("\n\n****************TERMINADO****************\n");
	fclose(arq);
	closedir(d);
	
	return 0;
}