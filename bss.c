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
	
	
	float *custos_gpu = NULL;
	int *caminho_gpu = NULL;
	int *capacidade_gpu = NULL;
	ADS *ads_gpu = NULL;

	alocarCustosGPU(fs, &custos_gpu);
	alocarSolucaoGPU (s, &caminho_gpu, &ads_gpu, &capacidade_gpu);
	printf("%d %.f\n", s.tamanhoCaminho, s.custo);
	runTest(s.tamanhoCaminho, g.n, s.custo, caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu);
   	CHECK_ERROR(cudaDeviceSynchronize());

	liberarCustosGPU(custos_gpu);
	liberarSolucaoGPU(caminho_gpu, ads_gpu);

	
	/*clock_t start, end;
	start = clock();
	Solucao s = instanciarSolucao(fs);
	end = clock();

	double t = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Construcao: %lf\n", t);

	//imprimirSolucao(s, fs);

	start = clock();
	Solucao ss = RVND(s, fs);
	end = clock();

	t = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("RVND: %lf\n", t);

	//imprimirSolucao(ss, fs);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(s);
	liberarSolucao(ss);*/

	return 0;
}