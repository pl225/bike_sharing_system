#include <string.h>
#include <math.h>

typedef struct FabricaSolucao
{
	int n;
	int q;
	int *demandas;
	int *verticesComDemanda;
	int numVerticesComDemanda;
	int numVerticesSemDemanda;
	int *verticesSemDemanda;
	float *custoArestas;
} FabricaSolucao;

typedef struct Solucao
{
	int *caminho;
	int tamanhoCaminho;
} Solucao;

void liberarFabrica(FabricaSolucao fs) {
	free(fs.demandas);
	free(fs.custoArestas);
	free(fs.verticesComDemanda);
	free(fs.verticesSemDemanda);
}

void liberarSolucao (Solucao s) {
	free(s.caminho);
}

int IndiceArestas(int i, int j, int n) {
	return i * n + j;
}

float distanciaEuclidiana (int i, int j, int * pontos) {
	int p1 = pontos[IndicePontos(i, 0)], p2 = pontos[IndicePontos(i, 1)],
		q1 = pontos[IndicePontos(j, 0)], q2 = pontos[IndicePontos(j, 1)];

	return sqrt(pow(q1 - p1, 2) + pow(q2 - p2, 2));
}

// https://stackoverflow.com/questions/6127503/shuffle-array-in-c
void shuffle(int *array, size_t n) {
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

FabricaSolucao instanciarFabrica (Grafo g) {
	FabricaSolucao fs;

	fs.q = g.q;
	fs.n = g.n;
	fs.demandas = (int*) malloc(sizeof(int) * fs.n);
	memcpy(fs.demandas, g.demandas, sizeof(int) * fs.n);
	fs.custoArestas = (float *) malloc(sizeof(float) * fs.n * fs.n);

	fs.verticesComDemanda = (int*) malloc(sizeof(int) * fs.n);
	fs.verticesSemDemanda = (int*) malloc(sizeof(int) * fs.n);
	int j = 0, k = 0; // j é o index dos vértices com demanda e k é o índice dos vértices sem demanda
	for (int i = 0; i < fs.n; i++) {
		if (fs.demandas[i] != 0) {
			fs.verticesComDemanda[j] = i;
			j++;
		} else {
			fs.verticesSemDemanda[k] = i;
			k++;
		}
	}

	fs.verticesComDemanda = (int*) realloc(fs.verticesComDemanda, sizeof(int) * (j));
	fs.numVerticesComDemanda = j;
	fs.verticesSemDemanda = (int*) realloc(fs.verticesSemDemanda, sizeof(int) * (k));
	fs.numVerticesSemDemanda = k;

	for (int i = 0; i < fs.n; i++) {
		for (j = 0; j < fs.n; j++) {
			if (i != j) {
				fs.custoArestas[IndiceArestas(i, j, fs.n)] = distanciaEuclidiana(i, j, g.pontos);
			} else {
				fs.custoArestas[IndiceArestas(i, j, fs.n)] = 0;
			}
		}
	}

	return fs;
}

Solucao instanciarSolucao (FabricaSolucao fs) {

	// -------------------- geração do vetor aleatório OV
	shuffle(fs.verticesComDemanda, fs.numVerticesComDemanda);
	shuffle(fs.verticesSemDemanda, fs.numVerticesSemDemanda);
	int verticesSemDemandaEscolhidos = fs.numVerticesSemDemanda > 0 ? rand() % fs.numVerticesSemDemanda : 0;

	const int tamanhoOV = fs.numVerticesComDemanda + verticesSemDemandaEscolhidos;
	int OV [tamanhoOV];
	memcpy(OV, fs.verticesComDemanda, sizeof(int) * fs.numVerticesComDemanda);

	for (int i = fs.numVerticesComDemanda, j = 0; tamanhoOV; i++, j++) 
		OV[i] = fs.verticesSemDemanda[j];

	// -------------------- fim da geração do vetor aleatório OV

	int tamanhoOVAux = tamanhoOV;
	Solucao solucao;

	while (tamanhoOVAux > 0) {

	}
}