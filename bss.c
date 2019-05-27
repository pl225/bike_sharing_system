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

int main(int argc, char *argv[])
{
	srand(time(NULL));
	Grafo g = carregarInstancia("instancias/n60q10A.tsp");
	FabricaSolucao fs = instanciarFabrica(g);

	Solucao x = GRASP(fs), xLinha, xLinhaLinha;

	const int kMax = 2;
	int k, maxIteracoes = 10 * g.n, i = 0;
	float melhorCusto = INFINITY;
	
	do {
		k = 1;

		do {
			xLinha = k == 0 ? _3OPT_P(x, fs) : splitP(x, fs);
			xLinhaLinha = RVND(xLinha, fs);

			if (xLinha.caminho != xLinhaLinha.caminho) liberarSolucao(xLinha);
			
			if (xLinhaLinha.custo < x.custo && isViavel(xLinhaLinha)) {
				liberarSolucao(x);
				x = xLinhaLinha;
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

	imprimirSolucao(x, fs);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(x);

	return 0;
}