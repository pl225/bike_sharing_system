#include <string.h>
#include <math.h>

#define TRUE 1
#define FALSE 0

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
	int *capacidades;
	int tamanhoCaminho;
	int viavel;
} Solucao;

void liberarFabrica(FabricaSolucao fs) { // mover
	free(fs.demandas);
	free(fs.custoArestas);
	free(fs.verticesComDemanda);
	free(fs.verticesSemDemanda);
}

void liberarSolucao (Solucao s) { // mover
	free(s.caminho);
}

int IndiceArestas(int i, int j, int n) { // mover
	return i * n + j;
}

float distanciaEuclidiana (int i, int j, int * pontos) { // mover
	int p1 = pontos[IndicePontos(i, 0)], p2 = pontos[IndicePontos(i, 1)],
		q1 = pontos[IndicePontos(j, 0)], q2 = pontos[IndicePontos(j, 1)];

	return sqrt(pow(q1 - p1, 2) + pow(q2 - p2, 2));
}

// https://stackoverflow.com/questions/6127503/shuffle-array-in-c
void shuffle(int *array, size_t n) { // mover
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

/*
	nº de vértices, vetor que possuirá as operações de valor máximo, vetor das demandas atuais, capacidade máxima, 
	capacidade atual, vetor que guardará os maiores índices de troca
*/
int computeTroca (int n, int troca[], int demandas [], int Q, int q, int indicesMaiorTroca[]) {

	int maiorTroca = 0, j = 0; // j guardará a maior quantidade de trocas que são maiores

	for (int i = 0; i < n; i++) {
		if (demandas[i] == 0) {
			troca[i] = 0;
			continue;
		}
		if (demandas[i] < 0) {
			troca[i] = q;
		} else {
			troca[i] = Q - q;
		}

		if (troca[i] > maiorTroca) {
			maiorTroca = troca[i];
			j = 0;
			indicesMaiorTroca[j] = i;
		} else if (troca[i] != 0 && troca[i] == maiorTroca) {
			j++;
			indicesMaiorTroca[j] = i;
		}
	}
	return j + 1;
}

int verticeMaisProximo (int n, int qtdIndicesMaiorTroca, int partida, float * custoArestas, int indicesMaiorTroca[]) {
	int maisProximo = indicesMaiorTroca[0], j;
	float menorDistancia = custoArestas[IndiceArestas(partida, maisProximo, n)];
	for (int i = 1; i < qtdIndicesMaiorTroca; i++) {
		j = indicesMaiorTroca[i]; // guarda o vértice candidato atual a ser o mais próximo
		if (custoArestas[IndiceArestas(partida, j, n)] < menorDistancia) {
			maisProximo = j;
			menorDistancia = custoArestas[IndiceArestas(partida, j, n)]; 
		}
	}
	return maisProximo;
}

Solucao instanciarSolucao (FabricaSolucao fs) {

	// -------------------- geração do vetor aleatório OV
	shuffle(fs.verticesComDemanda, fs.numVerticesComDemanda);
	shuffle(fs.verticesSemDemanda, fs.numVerticesSemDemanda);
	int verticesSemDemandaEscolhidos = fs.numVerticesSemDemanda > 0 ? rand() % fs.numVerticesSemDemanda : 0;

	const int tamanhoOV = fs.numVerticesComDemanda + verticesSemDemandaEscolhidos;
	int OV [tamanhoOV];
	memcpy(OV, fs.verticesComDemanda, sizeof(int) * fs.numVerticesComDemanda);

	for (int i = fs.numVerticesComDemanda, j = 0; i < tamanhoOV; i++, j++) 
		OV[i] = fs.verticesSemDemanda[j];

	// -------------------- fim da geração do vetor aleatório OV

	int tamanhoOVAux = tamanhoOV, q = fs.q; // iniciando q com todos os slots livres q = Q
	int demandas [fs.n], troca[fs.n], indicesMaiorTroca[fs.n];
	
	memcpy(demandas, fs.demandas, sizeof(int) * fs.n); // cópia de demandas para não sobrescrever a original
	
	Solucao solucao; // instanciação de uma nova solução
	solucao.caminho = (int *) malloc(sizeof(int) * fs.n);
	solucao.caminho[0] = 0;
	solucao.capacidades = (int *) malloc(sizeof(int) * fs.n);
	solucao.capacidades[0] = q;
	
	int j = 1, inserido, tamanhoAtualCaminho = fs.n;
	//demandas[0] = 0; //por enquanto, o depósito não possui a demanda prescrita por Pérez
	while (tamanhoOVAux > 0) {
		inserido = 0;
		for (int i = 0; i < tamanhoOV; i++) {
			if (OV[i] >= 0) { // se o vértice não tiver sido fechado
				// regra inversa de Pérez, mas de acordo com Adria
				if ((demandas[OV[i]] <= 0 && abs(demandas[OV[i]]) <= q) || (demandas[OV[i]] > 0 && fs.q - q >= demandas[OV[i]])) {
					
					solucao.caminho[j] = OV[i]; // adicionando o vértice ao caminho
					q += demandas[OV[i]]; // atualizando q
					solucao.capacidades[j] = q; // adicionando a capacidade entre o vertice j-1 e j
					j++;
					
					inserido = 1;
					demandas[OV[i]] = 0; // atualizando demanda do vértice i para zero

					// apagando o vértice na posição i de OV, isto é, ele foi fechado
					OV[i] = -1;
					tamanhoOVAux--;
					if (j == tamanhoAtualCaminho) { // caminho já chegou ao tamanho máximo
						tamanhoAtualCaminho += fs.n;
						solucao.caminho = (int*) realloc(solucao.caminho, sizeof(int) * tamanhoAtualCaminho);
						solucao.capacidades = (int*) realloc(solucao.capacidades, sizeof(int) * tamanhoAtualCaminho); 
					}
					break;
				}
			}
		}

		if (inserido == 0) {
			int qtdMaioresIndices = computeTroca(fs.n, troca, demandas, fs.q, q, indicesMaiorTroca);
			int maiorIndice;

			if (qtdMaioresIndices > 1) {
				int verticeAtual = solucao.caminho[j - 1];
				maiorIndice = verticeMaisProximo(fs.n, qtdMaioresIndices, verticeAtual, fs.custoArestas, indicesMaiorTroca);
			} else {
				maiorIndice = indicesMaiorTroca[0];
			}
			q += demandas[maiorIndice] > 0 ? troca[maiorIndice] : (-1) * troca[maiorIndice]; 
			demandas[maiorIndice] += demandas[maiorIndice] > 0 ? (-1) * troca[maiorIndice] : troca[maiorIndice]; 
			
			solucao.caminho[j] = maiorIndice;
			solucao.capacidades[j] = q;
			j++;
			if (j == tamanhoAtualCaminho) { // caminho já chegou ao tamanho máximo
				tamanhoAtualCaminho += fs.n;
				solucao.caminho = (int*) realloc(solucao.caminho, sizeof(int) * tamanhoAtualCaminho);
				solucao.capacidades = (int*) realloc(solucao.capacidades, sizeof(int) * tamanhoAtualCaminho); 
			}
		}
	}
	
	solucao.caminho = (int*) realloc(solucao.caminho, sizeof(int) * (j + 1));
	solucao.capacidades = (int*) realloc(solucao.capacidades, sizeof(int) * (j + 1));
	solucao.caminho[j] = 0;
	solucao.capacidades[j] = q;
	solucao.tamanhoCaminho = j + 1;
	solucao.viavel = TRUE;
	return solucao;
}