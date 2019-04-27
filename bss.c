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

	Grafo g = carregarInstancia(argv[1]);
	FabricaSolucao fs = instanciarFabrica(g);

	srand(time(NULL));

	return 0;
}