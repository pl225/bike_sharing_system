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

#ifndef CUDA_H
#define CUDA_H
#include "cuda.h"
#endif

#include <time.h>

#define GPU 0

int main(int argc, char *argv[])
{

	CHECK_ERROR(cudaSetDevice(GPU));
	CHECK_ERROR(cudaDeviceReset());

	//srand(time(NULL));

	Grafo g = carregarInstancia("instancias/n500q1000A.tsp");
	FabricaSolucao fs = instanciarFabrica(g);
	Solucao s = instanciarSolucao(fs);

	swap(s, fs);
	
	clock_t start, end;

	start = clock();
	Solucao ss = swap(s, fs);
	end = clock();

	double t = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Tempo: %lf\n", t);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(s);

	return 0;
}