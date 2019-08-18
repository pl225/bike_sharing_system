#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#ifndef VIZINHANCA_H
#define VIZINHANCA_H
#include "vizinhanca.h"
#endif

#define CHECK_ERROR(call) do {                                                    \
    if(cudaSuccess != call) {                                                     \
        fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", cudaGetErrorString(call),  __FILE__, __LINE__); \
        exit(0);                                                                                 \
    } } while (0)

#include <stdio.h>

typedef enum
{
	_2OPT_GPU, SWAP_GPU, SPLIT_GPU, OROPT1_GPU, OROPT2_GPU, OROPT3_GPU, OROPT4_GPU	
} Vizinhanca;

typedef struct Reduzido
{
	int i, j;
	float custo;
} Reduzido;

void alocarCustosGPU (FabricaSolucao fs, float** custos_gpu);

void liberarCustosGPU (float* custos_gpu);

void alocarSolucaoGPU (Solucao s, int** caminho_gpu, ADS** ads_gpu, int** capacidade_gpu);

void liberarSolucaoGPU (int* caminho_gpu, ADS* ads_gpu, int* capacidade_gpu);

Reduzido obterVizinho (Solucao s, FabricaSolucao fs, float* custos_gpu, Vizinhanca v);