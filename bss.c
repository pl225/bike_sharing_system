#ifndef GRAFO_H
#define GRAFO_H
#include "grafo.h"
#endif

#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	srand(time(NULL));

	Grafo g = carregarInstancia("instancias/n20q10A.tsp");
	FabricaSolucao fs = instanciarFabrica(g);

	Solucao s = GRASP(fs);

	imprimirSolucao(s, fs);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(s);

	return 0;
}