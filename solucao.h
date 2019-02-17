#ifndef GRAFO_H
#define GRAFO_H
#include "grafo.h"
#endif

typedef struct FabricaSolucao
{
	int n;
	int q;
	int *demandas; // signed char
	int *verticesComDemanda; // unsigned char
	int numVerticesComDemanda;
	int numVerticesSemDemanda;
	int *verticesSemDemanda; // unsigned char
	float *custoArestas;
} FabricaSolucao;

typedef struct ADS
{
	short qSum, qMin, qMax, lMin, lMax;
} ADS;

typedef struct Solucao
{
	int *caminho; // unsigned char
	int *capacidades; // signed char
	ADS **ads;
	int tamanhoCaminho;
	float custo;
} Solucao;

int IndiceArestas(int i, int j, int n);

void liberarFabrica(FabricaSolucao fs);

void liberarSolucao (Solucao s);

float custo (Solucao s, FabricaSolucao fs);

int isViavel (Solucao s);

void imprimirSolucao (Solucao s, FabricaSolucao fs);

Solucao copiarSolucao (Solucao s);

void construirADS (Solucao s, int q);

void atualizarADS (Solucao s, int q, int inicio, int fim);

FabricaSolucao instanciarFabrica (Grafo g);

int* construirOV_ILS_RVND (FabricaSolucao fs, int* tamanhoOV);

int* construirOV_Greedy (FabricaSolucao fs, int* tamanhoOV);

int escolherProximoVertice_ILS_RVND (int OV[], int demandas[], int tamanhoOV, FabricaSolucao fs, int q, int estacaoAnterior);

int escolherProximoVertice_Greedy (int OV[], int demandas[], int tamanhoOV, FabricaSolucao fs, int q, int estacaoAnterior);

Solucao instanciarSolucao (FabricaSolucao fs, int* (*construcaoOV)(FabricaSolucao, int*), int (*escolherProximoVertice)(int [], int [], int, FabricaSolucao, int, int));