#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "solucao.h"

void liberarFabrica(FabricaSolucao fs) {
	free(fs.demandas);
	free(fs.custoArestas);
	free(fs.verticesComDemanda);
	free(fs.verticesSemDemanda);
}

void liberarSolucao (Solucao s) { 
	free(s.caminho);
	free(s.capacidades);
	for (int i = 0; i < s.tamanhoCaminho; i++) free(s.ads[i]);
	free(s.ads);
}

int IndiceArestas(int i, int j, int n) {
	return i * n + j;
}

float distanciaEuclidiana (int i, int j, int * pontos) { // mover
	int p1 = pontos[IndicePontos(i, 0)], p2 = pontos[IndicePontos(i, 1)],
		q1 = pontos[IndicePontos(j, 0)], q2 = pontos[IndicePontos(j, 1)];

	return floor(sqrt(pow(q1 - p1, 2) + pow(q2 - p2, 2)));
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

float custo (Solucao s, FabricaSolucao fs) { // mover
	float f = 0;
	for (int i = 0; i < s.tamanhoCaminho - 1; i++) {
		f += fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
	}
	return f;
}

int isViavel (Solucao s) {
	return s.ads[0][s.tamanhoCaminho - 1].lMin == 0 && s.ads[0][s.tamanhoCaminho - 1].lMax >= 0;
}

void imprimirSolucao (Solucao s, FabricaSolucao fs) {
	printf("\nImprimindo solução\nSituação: ");
	if (isViavel(s)) 
		printf("viável\n");
	else
		printf("inviável\n");
	printf("\nCusto: %.f, custo(s, fs): %.f, tamanho do caminho: %d\n", s.custo, custo(s, fs), s.tamanhoCaminho);
	printf("Caminho:\n\t");
	for (int i = 0; i < s.tamanhoCaminho; i++) printf("%d ", s.caminho[i] + 1);
	printf("\nCapacidades: \n\t");
	for (int i = 0; i < s.tamanhoCaminho; i++) printf("%d ", s.capacidades[i]);
	printf("\n");
}

Solucao copiarSolucao (Solucao s) { // mover
	Solucao copia;
	copia.tamanhoCaminho = s.tamanhoCaminho;

	size_t tamanhoInteiroTotal = sizeof(int) * s.tamanhoCaminho;

	copia.caminho = (int *) malloc(tamanhoInteiroTotal);
	copia.capacidades = (int *) malloc(tamanhoInteiroTotal);
	copia.custo = s.custo;
	memcpy(copia.caminho, s.caminho, tamanhoInteiroTotal);
	memcpy(copia.capacidades, s.capacidades, tamanhoInteiroTotal);

	size_t tamanhoADS = sizeof(ADS) * s.tamanhoCaminho;

	copia.ads = (ADS **) malloc(sizeof(ADS *) * copia.tamanhoCaminho);
	for (int i = 0; i < copia.tamanhoCaminho; i++) {
		copia.ads[i] = (ADS*) malloc(tamanhoADS);
		memcpy(copia.ads[i], s.ads[i], tamanhoADS);
	}
	return copia;
}

void construirADS (Solucao s, int q) {
	short qSumAuxiliar, cargaMinima, cargaMaxima;
	for (int i = 0; i < s.tamanhoCaminho; i++) {
			
		qSumAuxiliar = i != 0 ? s.capacidades[i - 1] - s.capacidades[i] : 0;
		cargaMinima = qSumAuxiliar < 0 ? qSumAuxiliar : 0;
		cargaMaxima = qSumAuxiliar > 0 ? qSumAuxiliar : 0;
		
		s.ads[i][i].qSum = qSumAuxiliar;
		s.ads[i][i].qMin = cargaMinima;
		s.ads[i][i].qMax = cargaMaxima;
		s.ads[i][i].lMin = -s.ads[i][i].qMin;
		s.ads[i][i].lMax = q - s.ads[i][i].qMax;
		
		for (int j = i + 1; j < s.tamanhoCaminho; j++) {
			
			qSumAuxiliar += s.capacidades[j - 1] - s.capacidades[j];
			if (qSumAuxiliar < cargaMinima) cargaMinima = qSumAuxiliar;
			if (qSumAuxiliar > cargaMaxima) cargaMaxima = qSumAuxiliar;
			
			s.ads[i][j].qSum = qSumAuxiliar;
			s.ads[i][j].qMin = cargaMinima;
			s.ads[i][j].qMax = cargaMaxima;
			s.ads[i][j].lMin = -s.ads[i][j].qMin;
			s.ads[i][j].lMax = q - s.ads[i][j].qMax;
		}
	}
}

void atualizarADS (Solucao s, int q, int inicio, int fim) { // inicio sempre > que zero
	short qSumAuxiliar, cargaMinima, cargaMaxima;
	int aux = inicio - 1, inicioLoop2;
	for (int i = 0; i <= fim; i++) { // o índice de fim deve ser incluído
		
		if (aux >= i) {	
			qSumAuxiliar = s.ads[i][aux].qSum + (s.capacidades[aux] - s.capacidades[inicio]);
			cargaMinima = qSumAuxiliar < s.ads[i][aux].qMin ? qSumAuxiliar : s.ads[i][aux].qMin;
			cargaMaxima = qSumAuxiliar > s.ads[i][aux].qMax ? qSumAuxiliar : s.ads[i][aux].qMax;
			inicioLoop2 = inicio;
		} else {
			if (i > 0) qSumAuxiliar = s.capacidades[i - 1] - s.capacidades[i];
			else qSumAuxiliar = 0;
			cargaMinima = qSumAuxiliar < 0 ? qSumAuxiliar : 0;
			cargaMaxima = qSumAuxiliar > 0 ? qSumAuxiliar : 0;
			inicioLoop2 = i;
		}
		s.ads[i][inicioLoop2].qSum = qSumAuxiliar;
		s.ads[i][inicioLoop2].qMin = cargaMinima;
		s.ads[i][inicioLoop2].qMax = cargaMaxima;
		s.ads[i][inicioLoop2].lMin = -s.ads[i][inicioLoop2].qMin;
		s.ads[i][inicioLoop2].lMax = q - s.ads[i][inicioLoop2].qMax;
			
		for (int j = inicioLoop2 + 1; j < s.tamanhoCaminho; j++) {
			
			qSumAuxiliar += s.capacidades[j - 1] - s.capacidades[j];
			if (qSumAuxiliar < cargaMinima) cargaMinima = qSumAuxiliar;
			if (qSumAuxiliar > cargaMaxima) cargaMaxima = qSumAuxiliar;
			
			s.ads[i][j].qSum = qSumAuxiliar;
			s.ads[i][j].qMin = cargaMinima;
			s.ads[i][j].qMax = cargaMaxima;
			s.ads[i][j].lMin = -s.ads[i][j].qMin;
			s.ads[i][j].lMax = q - s.ads[i][j].qMax;
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

float avaliarCustoInsercaoVertice (FabricaSolucao fs, int LC[], int demandas[], int q, int indiceVertice, int ultimoVertice) {
	
	float custo = fs.custoArestas[IndiceArestas(ultimoVertice, LC[indiceVertice], fs.n)], fluxoMax = 0;

	int demandaVertice = demandas[LC[indiceVertice]];
	
	/*for (int i = 0; i < fs.numVerticesComDemanda; i++)
		if (LC[i] >= 0 && LC[i] != LC[indiceVertice] && demandas[LC[i]] != 0)
			custo += fs.custoArestas[IndiceArestas(LC[indiceVertice], LC[i], fs.n)];*/

	float y = (rand() % 171) / 100.f;
	
	if (demandaVertice < 0)
		fluxoMax = abs(demandaVertice) <= q ? abs(demandaVertice) : q;
	else
		fluxoMax = fs.q - q >= demandaVertice ? demandaVertice : fs.q - q;

	if (fluxoMax > 0) return custo - y * fluxoMax;
	else return INFINITY;
}

/*
	FabricaSolucao, vetor que terá os custos, vetor dos candidatos atuais, demandas atuais dos candidatos, 
	capacidade atual, último vértice visitado
*/
void avaliarCustoInsercao(FabricaSolucao fs, float g[], int LC[], int demandas[], int q, int ultimoVertice, float *custoMin, float *custoMax) {

	*custoMin = INFINITY, *custoMax = 0;

	for (int i = 0; i < fs.numVerticesComDemanda; i++) {
		if (LC[i] >= 0 && LC[i] != ultimoVertice) {
			g[i] = avaliarCustoInsercaoVertice(fs, LC, demandas, q, i, ultimoVertice);
			if (g[i] <= *custoMin && g[i] != INFINITY) *custoMin = g[i];
			if (g[i] >= *custoMax && g[i] != INFINITY) *custoMax = g[i];
		} else {
			g[i] = INFINITY;
		}
	}
}

int construirListaRestritaDeCandidatos (int LC[], float g[], int LRC[], float custoMin, float custoMax, int n) {
	float alpha = (float) rand() / (float) RAND_MAX;
	float limite = custoMin + alpha * (custoMax - custoMin);
	int a = 0;
	
	for (int i = 0; i < n; i++) {
		if (LC[i] >= 0 && g[i] <= limite) {
			LRC[a] = i;
			a++;
		}
	}

	return a;
}

int atualizarDemandaVertice(int e, int demandas[], int q, int LC[], int capMax, int *tamanhoLC) {

	if ((demandas[LC[e]] < 0 && abs(demandas[LC[e]]) <= q) || (demandas[LC[e]] > 0 && capMax - q >= demandas[LC[e]])) {
		q += demandas[LC[e]];
		demandas[LC[e]] = 0;
		LC[e] = -1;
		*tamanhoLC = *tamanhoLC - 1;
	} else if (demandas[LC[e]] < 0) {
		demandas[LC[e]] += q;
		q = 0;
	} else {
		demandas[LC[e]] -= (capMax - q);
		q = capMax;
	}

	return q;
}

Solucao GRASP (FabricaSolucao fs) {
	
	int demandas[fs.n], q = fs.q;
	memcpy(demandas, fs.demandas, sizeof(int) * fs.n); // cópia de demandas para não sobrescrever a original

	Solucao solucao; // instanciação de uma nova solução
	solucao.caminho = malloc(sizeof(int) * fs.n);
	solucao.caminho[0] = 0;
	solucao.capacidades = malloc(sizeof(int) * fs.n);
	solucao.capacidades[0] = q;
	solucao.custo = 0;

	int LC[fs.numVerticesComDemanda], LRC[fs.numVerticesComDemanda]; // iniciando a lista de candidatos
	float g[fs.numVerticesComDemanda], custoMin, custoMax; // conterá os custos de inserção
	memcpy(LC, fs.verticesComDemanda, sizeof(int) * fs.numVerticesComDemanda);
	
	int tamanhoLC = fs.numVerticesComDemanda, j = 1, e, qtdLRC, tamanhoAtualCaminho = fs.n;

	while (tamanhoLC > 0) {
		
		avaliarCustoInsercao(fs, g, LC, demandas, q, solucao.caminho[j - 1], &custoMin, &custoMax);

		qtdLRC = construirListaRestritaDeCandidatos(LC, g, LRC, custoMin, custoMax, fs.numVerticesComDemanda);
		e = LRC[rand() % qtdLRC];

		solucao.caminho[j] = LC[e]; // adicionando o vértice ao caminho
		solucao.custo += fs.custoArestas[IndiceArestas(solucao.caminho[j - 1], LC[e], fs.n)];

		q = atualizarDemandaVertice(e, demandas, q, LC, fs.q, &tamanhoLC);

		solucao.capacidades[j] = q;
		j++;

		if (j == tamanhoAtualCaminho) { // caminho já chegou ao tamanho máximo
			tamanhoAtualCaminho += fs.n;
			solucao.caminho = realloc(solucao.caminho, sizeof(int) * tamanhoAtualCaminho);
			solucao.capacidades = realloc(solucao.capacidades, sizeof(int) * tamanhoAtualCaminho); 
		}
		
	}

	solucao.caminho = realloc(solucao.caminho, sizeof(int) * (j + 1));
	solucao.capacidades = realloc(solucao.capacidades, sizeof(int) * (j + 1));
	solucao.caminho[j] = 0;
	solucao.custo += fs.custoArestas[IndiceArestas(solucao.caminho[j - 1], 0, fs.n)];
	solucao.capacidades[j] = q;
	solucao.tamanhoCaminho = j + 1;

	solucao.ads = (ADS**) malloc(sizeof(ADS*) * solucao.tamanhoCaminho);
	for (int i = 0; i < solucao.tamanhoCaminho; i++) solucao.ads[i] = (ADS*) malloc(sizeof(ADS) * solucao.tamanhoCaminho);
	
	construirADS(solucao, fs.q);

	return solucao;
}