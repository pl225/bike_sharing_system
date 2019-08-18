#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#ifndef VIZINHANCA_H
#define VIZINHANCA_H
#include "vizinhanca.h"
#endif

#include <stdio.h>

#define CHECK_ERROR(call) do {                                                    \
	if(cudaSuccess != call) {                                                     \
		fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", cudaGetErrorString(call),  __FILE__, __LINE__); \
		exit(0);                                                                                 \
    } } while (0)

typedef struct Reduzido
{
	int i, j;
	float custo;
} Reduzido;

void alocarCustosGPU (FabricaSolucao fs, float** custos_gpu);

void liberarCustosGPU (float* custos_gpu);

void alocarSolucaoGPU (Solucao s, int** caminho_gpu, ADS** ads_gpu, int** capacidade_gpu);

void liberarSolucaoGPU (int* caminho_gpu, ADS* ads_gpu);

void runTest(int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, int* capacidade_gpu);