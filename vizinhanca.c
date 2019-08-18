#include "vizinhanca.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

void mergeAux (Solucao *s, int destino, int origem, int tamanhoCaminho) {
	size_t tamPedacoMovido = tamanhoCaminho - origem, tamanhoADS = sizeof(ADS) * tamPedacoMovido;

	memcpy(s->caminho + destino, s->caminho + origem, sizeof(int) * tamPedacoMovido);
	memcpy(s->capacidades + destino, s->capacidades + origem, sizeof(int) * tamPedacoMovido);
	free(s->ads[destino]);
	memcpy(s->ads + destino, s->ads + origem, sizeof(ADS*) * tamPedacoMovido);
	
	for (int j = 0; j < tamanhoCaminho; j++) {
		memcpy(s->ads[j] + destino, s->ads[j] + origem, tamanhoADS);
	}
}


void merge(Solucao *s, int Q, int indiceTrocaI, int indiceTrocaJ) {
	int i = indiceTrocaI, j = indiceTrocaJ, tamanhoCaminho = s->tamanhoCaminho;
	
	if (indiceTrocaI > 1 && s->caminho[i - 1] == s->caminho[i]) {
		mergeAux(s, i - 1, i, tamanhoCaminho);
		i -= 1, j -= 1, tamanhoCaminho -= 1;
	}
	if (s->caminho[i] == s->caminho[i + 1]) {
		mergeAux(s, i, i + 1, tamanhoCaminho);
		j -= 1, tamanhoCaminho -= 1;
	}
	if (s->caminho[j - 1] == s->caminho[j]) {
		mergeAux(s, j - 1, j, tamanhoCaminho);
		j -= 1, tamanhoCaminho -= 1;
	}
	if (indiceTrocaJ < s->tamanhoCaminho - 1 && s->caminho[j] == s->caminho[j + 1]) {
		mergeAux(s, j, j + 1, tamanhoCaminho);
		tamanhoCaminho -= 1;
	}

	size_t tamanhoADS = sizeof(ADS) * tamanhoCaminho;
	s->caminho = (int *) realloc(s->caminho, sizeof(int) * tamanhoCaminho);
	s->capacidades = (int *) realloc(s->capacidades, sizeof(int) * tamanhoCaminho);
	s->ads = (ADS**) realloc(s->ads, sizeof(ADS*) * tamanhoCaminho);
	
	for (int k = 0; k < tamanhoCaminho; k++) s->ads[k] = (ADS*) realloc(s->ads[k], tamanhoADS);
	s->tamanhoCaminho = tamanhoCaminho;
	atualizarADS(*s, Q, i, j);
}

Solucao swap(Solucao s, FabricaSolucao fs, float* custos_gpu) {

	Reduzido r = obterVizinho(s, fs, custos_gpu, SWAP_GPU);

	int indiceTrocaI = r.i, indiceTrocaJ = r.j;
	float menorCusto = r.custo;

	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		int aux = nova.caminho[indiceTrocaI];
		nova.caminho[indiceTrocaI] = nova.caminho[indiceTrocaJ];
		nova.caminho[indiceTrocaJ] = aux;
		nova.custo = menorCusto;

		int qI = nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaI - 1];
		int qJ = nova.capacidades[indiceTrocaJ] - nova.capacidades[indiceTrocaJ - 1];

		if (qI != qJ) {
			int qImais1 = nova.capacidades[indiceTrocaI + 1] - nova.capacidades[indiceTrocaI], aux;
			nova.capacidades[indiceTrocaI] = nova.capacidades[indiceTrocaI - 1] + qJ;
			for (int i = indiceTrocaI + 1; i < indiceTrocaJ; i++) {
				aux = qImais1;
				qImais1 = nova.capacidades[i + 1] - nova.capacidades[i];
				nova.capacidades[i] = nova.capacidades[i - 1] + aux;
			}
		}
		merge(&nova, fs.q, indiceTrocaI, indiceTrocaJ);

		return nova;
	} else {
		return s;
	}
}

Solucao orOPT(Solucao s, FabricaSolucao fs, float* custos_gpu, int tipo) {

	Reduzido r;
	int passo;

	if (tipo == 0) { // reinsercao
		passo = 0;
		r = obterVizinho(s, fs, custos_gpu, OROPT1_GPU);
	} else if (tipo == 1) { // orOPT2
		passo = 1;
	 	r = obterVizinho(s, fs, custos_gpu, OROPT2_GPU);
	} else if (tipo == 2) { // orOPT3
		passo = 2;
	 	r = obterVizinho(s, fs, custos_gpu, OROPT3_GPU);
	} else { // orOPT4
		passo = 3;
		r = obterVizinho(s, fs, custos_gpu, OROPT4_GPU);
	}

	int indiceTrocaI = r.i, indiceTrocaJ = r.j;
	float menorCusto = r.custo;

	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		nova.custo = menorCusto;
		int verticesMovidos[passo + 1], capacidadesMovidas[passo + 1];

		memcpy(verticesMovidos, nova.caminho + indiceTrocaI, sizeof(int) * (passo + 1));
		memcpy(capacidadesMovidas, nova.capacidades + indiceTrocaI, sizeof(int) * (passo + 1));
		if (indiceTrocaI < indiceTrocaJ) {
			int posAux = indiceTrocaI + passo + 1;
			int trazidosTras = indiceTrocaJ - posAux + 1;
			memcpy(nova.caminho + indiceTrocaI, nova.caminho + posAux, sizeof(int) * trazidosTras);
			memcpy(nova.caminho + indiceTrocaI + trazidosTras, verticesMovidos, sizeof(int) * (passo + 1));

			int diff = nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaI - 1];
			int inicioTrocaCapacidade = indiceTrocaI + trazidosTras;

			for (int i = indiceTrocaI, a = posAux, k = 0; k < trazidosTras; i++, a++, k++) {
				nova.capacidades[i] = nova.capacidades[i - 1] + (nova.capacidades[a] - nova.capacidades[a - 1]);
			}

			nova.capacidades[inicioTrocaCapacidade] = nova.capacidades[inicioTrocaCapacidade - 1] + diff;
			for (int i = inicioTrocaCapacidade + 1, a = 1; a < passo + 1; i++, a++) {
				nova.capacidades[i] = nova.capacidades[i - 1] + (capacidadesMovidas[a] - capacidadesMovidas[a - 1]);
			} 
			merge(&nova, fs.q, indiceTrocaI, indiceTrocaJ);
		} else {
			int posAux = indiceTrocaJ + passo + 1;
			int trazidosFrente = indiceTrocaI - indiceTrocaJ;
			
			memcpy(nova.caminho + posAux, nova.caminho + indiceTrocaJ, sizeof(int) * trazidosFrente); //modificação do caminho
			memcpy(nova.caminho + indiceTrocaJ + 1, verticesMovidos, sizeof(int) * (passo + 1));

			int diff = nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaI - 1];
			int operacoesFrente[trazidosFrente - 1];

			for (int i = 0, a = indiceTrocaJ + 1; i < trazidosFrente - 1; i++, a++) {
				operacoesFrente[i] = nova.capacidades[a] - nova.capacidades[a - 1];
			}

			nova.capacidades[indiceTrocaJ + 1] = nova.capacidades[indiceTrocaJ] + diff;
			for (int i = indiceTrocaJ + 2, a = 1, k = indiceTrocaI + 1; a < passo + 1; i++, a++, k++) {
				nova.capacidades[i] = nova.capacidades[i - 1] + (nova.capacidades[k] - nova.capacidades[k - 1]);
			}

			for (int i = posAux + 1, a = 0; a < trazidosFrente - 1; i++, a++) {
				nova.capacidades[i] = nova.capacidades[i - 1] + operacoesFrente[a];
			}

			merge(&nova, fs.q, indiceTrocaJ + 1, indiceTrocaI + passo);	
		}
	
		return nova;
	} else {
		return s;
	}
}

Solucao reinsercao(Solucao s, FabricaSolucao fs, float* custos_gpu) {
	return orOPT(s, fs, custos_gpu, 0);
}

Solucao orOPT2(Solucao s, FabricaSolucao fs, float* custos_gpu) {
	return orOPT(s, fs, custos_gpu, 1);
}

Solucao orOPT3(Solucao s, FabricaSolucao fs, float* custos_gpu) {
	return orOPT(s, fs, custos_gpu, 2);
}

Solucao orOPT4(Solucao s, FabricaSolucao fs, float* custos_gpu) {
	return orOPT(s, fs, custos_gpu, 3);
}

Solucao _2OPT (Solucao s, FabricaSolucao fs, float* custos_gpu) {

	Reduzido r = obterVizinho(s, fs, custos_gpu, _2OPT_GPU);

	int indiceTrocaI = r.i, indiceTrocaJ = r.j, aux;
	float menorCusto = r.custo;

	if (indiceTrocaI != - 1) {
		Solucao copia = copiarSolucao(s);
		copia.custo = menorCusto;
		for (int a = indiceTrocaI + 1, b = indiceTrocaJ - 1; a < b; a++, b--) { // reversão entre i e j
			aux = copia.caminho[a];
			copia.caminho[a] = copia.caminho[b];
			copia.caminho[b] = aux; 
		}
		short diff;
		for (int a = indiceTrocaI + 1, b = indiceTrocaJ - 1; a < indiceTrocaJ; a++, b--) {
			diff = s.capacidades[b] - s.capacidades[b - 1];
			copia.capacidades[a] = copia.capacidades[a - 1] + diff;
		}
		merge(&copia, fs.q, indiceTrocaI + 1, indiceTrocaJ - 1);		
		return copia;
	} else {
		return s;
	}
}

Solucao autalizacaoParaSplit (Solucao s, FabricaSolucao fs, int indiceTrocaI, int indiceTrocaJ, float novoCusto) {
	Solucao copia = copiarSolucao(s);
	copia.tamanhoCaminho += 1;
	copia.caminho = (int *) realloc(copia.caminho, sizeof(int) * copia.tamanhoCaminho);
	copia.capacidades = (int *) realloc(copia.capacidades, sizeof(int) * copia.tamanhoCaminho);
	copia.custo = novoCusto;

	int aux = copia.caminho[indiceTrocaI];
	memcpy(copia.caminho + indiceTrocaJ + 1, copia.caminho + indiceTrocaJ, sizeof(int) * (s.tamanhoCaminho - indiceTrocaJ));
	copia.caminho[indiceTrocaJ] = aux;

	int inicioLoop, fimLoop, fatorAlteracao;

	if (indiceTrocaI < indiceTrocaJ) {
		fatorAlteracao = copia.capacidades[indiceTrocaI] - copia.capacidades[indiceTrocaI - 1] < 0 ? 1 : -1;
		inicioLoop = indiceTrocaI, fimLoop = indiceTrocaJ;
	} else {
		inicioLoop = indiceTrocaJ, fimLoop = indiceTrocaI + 1;
		fatorAlteracao = copia.capacidades[indiceTrocaI] - copia.capacidades[indiceTrocaI - 1] < 0 ? -1 : 1;
	}

	aux = copia.capacidades[indiceTrocaI];
	memcpy(copia.capacidades + indiceTrocaJ + 1, copia.capacidades + indiceTrocaJ, sizeof(int) * (s.tamanhoCaminho - indiceTrocaJ));
	copia.capacidades[indiceTrocaJ] = aux;

	for (int a = inicioLoop; a < fimLoop; a++) copia.capacidades[a] += fatorAlteracao;

	if (indiceTrocaI < indiceTrocaJ)
		fatorAlteracao *= -1;
	copia.capacidades[indiceTrocaJ] = copia.capacidades[indiceTrocaJ - 1] + fatorAlteracao;

	copia.ads = (ADS**) realloc(copia.ads, sizeof(ADS*) * copia.tamanhoCaminho);
	memcpy(copia.ads + indiceTrocaJ + 1, copia.ads + indiceTrocaJ, sizeof(ADS*) * (s.tamanhoCaminho - indiceTrocaJ));
	copia.ads[indiceTrocaJ] = (ADS*) malloc(sizeof(ADS) * copia.tamanhoCaminho);

	size_t tamanhoADS = sizeof(ADS) * (s.tamanhoCaminho - indiceTrocaJ);
	for (int i = 0; i < copia.tamanhoCaminho; i++) {
		if (i != indiceTrocaJ) {
			copia.ads[i] = (ADS*) realloc(copia.ads[i], sizeof(ADS) * copia.tamanhoCaminho);
			memcpy(copia.ads[i] + indiceTrocaJ + 1, copia.ads[i] + indiceTrocaJ, tamanhoADS);
		}
	}

	atualizarADS(copia, fs.q, inicioLoop, fimLoop);

	return copia;
}

Solucao split (Solucao s, FabricaSolucao fs, float* custos_gpu) {

	Reduzido r = obterVizinho(s, fs, custos_gpu, SPLIT_GPU);

	int indiceTrocaI = r.i, indiceTrocaJ = r.j;
	float menorCusto = r.custo;

	if (indiceTrocaI != -1) {
		return autalizacaoParaSplit(s, fs, indiceTrocaI, indiceTrocaJ, menorCusto);
	} else {
		return s;
	}
}

Solucao RVND (Solucao s, FabricaSolucao fs, float* custos_gpu) {
	Solucao melhorSolucao = copiarSolucao(s), sLinha;
	int indices[] = {0, 1, 2, 3, 4, 5, 6};
	Solucao (*vizinhancas[])(Solucao, FabricaSolucao, float*) = {split, reinsercao, _2OPT, orOPT2, orOPT3, orOPT4, swap};
	int LN = 7, N, aux;
	while (LN > 0) {
		N = rand() % LN;
		sLinha = (*vizinhancas[indices[N]])(melhorSolucao, fs, custos_gpu);
		if (sLinha.custo < melhorSolucao.custo) {
			liberarSolucao(melhorSolucao);
			melhorSolucao = sLinha;
			LN = 7;
		} else {
			if (sLinha.caminho != melhorSolucao.caminho) liberarSolucao(sLinha);
			LN--;
			if (N < 6) {
				aux = indices[N];
				memcpy(indices + N, indices + N + 1, sizeof(int) * (6 - N));
				indices[6] = aux;
			}
		}
	}
	if (melhorSolucao.custo == s.custo) {
		liberarSolucao(melhorSolucao);
		return s;
	}
	return melhorSolucao;
}

Solucao doubleBridge (Solucao s, FabricaSolucao fs) {
	Solucao copia = copiarSolucao(s);
	int tamanho = copia.tamanhoCaminho - 1;
	const int tamInt = sizeof(int);
	int p1 = 1 + rand() % (tamanho - 5),
		p2 = rand() % ((tamanho - 4) - p1) + p1,
		p3 = rand() % ((tamanho - 3) - p2) + p2 + 1,
		p4 = rand() % ((tamanho - 2 - p3)) + p3;
	
	int tamSecao1 = p2 - p1 + 1, secao1[tamSecao1], tamSecao2 = (p3 - 1) - (p2 + 1) + 1, tamSecao3 = p4 - p3 + 1;
	memcpy(secao1, copia.caminho + p1, tamSecao1 * tamInt);

	copia.custo = s.custo - (fs.custoArestas[IndiceArestas(s.caminho[p1 - 1], s.caminho[p1], fs.n)]
							+ fs.custoArestas[IndiceArestas(s.caminho[p4], s.caminho[p4 + 1], fs.n)]);
	copia.custo += (fs.custoArestas[IndiceArestas(s.caminho[p1 - 1], s.caminho[p3], fs.n)]
					+ fs.custoArestas[IndiceArestas(s.caminho[p2], s.caminho[p4 + 1], fs.n)]);
	
	if (tamSecao2 == 0) {

		copia.custo -= fs.custoArestas[IndiceArestas(s.caminho[p2], s.caminho[p3], fs.n)];
		copia.custo += fs.custoArestas[IndiceArestas(s.caminho[p4], s.caminho[p1], fs.n)];

		memcpy(copia.caminho + p1, copia.caminho + p3, tamSecao3 * tamInt);
		memcpy(copia.caminho + p1 + tamSecao3, secao1, tamSecao1 * tamInt);
	} else {

		copia.custo -= (fs.custoArestas[IndiceArestas(s.caminho[p2], s.caminho[p2 + 1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p3 - 1], s.caminho[p3], fs.n)]);
		copia.custo += (fs.custoArestas[IndiceArestas(s.caminho[p4], s.caminho[p2 + 1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p3 - 1], s.caminho[p1], fs.n)]);

		int secao2[tamSecao2];
		memcpy(secao2, copia.caminho + p2 + 1, tamSecao2 * tamInt);
		memcpy(copia.caminho + p1, copia.caminho + p3, tamSecao3 * tamInt);
		memcpy(copia.caminho + p1 + tamSecao3, secao2, tamSecao2 * tamInt);
		memcpy(copia.caminho + p1 + tamSecao2 + tamSecao3, secao1, tamSecao1 * tamInt);
	}

	int capSecao1[tamSecao1];
	int capIni1 = copia.capacidades[p1] - copia.capacidades[p1 - 1];
	memcpy(capSecao1, copia.capacidades + p1, tamSecao1 * tamInt);
	int *capSecao2 = NULL, capIni2 = - 1;
	if (tamSecao2 > 0) {
		capSecao2 = (int *) malloc(tamSecao2 * tamInt);
		memcpy(capSecao2, copia.capacidades + p2 + 1, tamSecao2 * tamInt);
		capIni2 = copia.capacidades[p2 + 1] - copia.capacidades[p2];
	}
	copia.capacidades[p1] = copia.capacidades[p1 - 1] + (copia.capacidades[p3] - copia.capacidades[p3 - 1]);
	for (int i = p1 + 1, a = p3 + 1; i < p1 + tamSecao3; i++, a++)
		copia.capacidades[i] = copia.capacidades[i - 1] + (copia.capacidades[a] - copia.capacidades[a - 1]);
	if (tamSecao2 > 0) {
		copia.capacidades[p1 + tamSecao3] = copia.capacidades[p1 + tamSecao3 - 1] + capIni2;
		for (int i = p1 + tamSecao3 + 1, a = 1; i < p1 + tamSecao3 + tamSecao2; i++, a++)
			copia.capacidades[i] = copia.capacidades[i - 1] + (capSecao2[a] - capSecao2[a - 1]);
		free(capSecao2);
	}
	copia.capacidades[p1 + tamSecao3 + tamSecao2] = copia.capacidades[p1 + tamSecao3 + tamSecao2 - 1] + capIni1;
	for (int i = p1 + tamSecao3 + tamSecao2 + 1, a = 1; i < p1 + tamSecao3 + tamSecao2 + tamSecao1; i++, a++)
		copia.capacidades[i] = copia.capacidades[i - 1] + (capSecao1[a] - capSecao1[a - 1]);

	merge(&copia, fs.q, p1, p4);

	return copia;
}

Solucao splitP (Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = s.custo, custoParcial;
	int indiceTrocaI = -1, indiceTrocaJ = -1;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		if (fs.demandas[s.caminho[i]] < - 1 || fs.demandas[s.caminho[i]] > 1) {
			for (int j = 1; j < s.tamanhoCaminho - 1; j++) {

				if (i == j) continue;
				if (s.caminho[i] == s.caminho[j] || s.caminho[i] == s.caminho[j - 1] || s.caminho[j] == s.caminho[i - 1]) continue;
				if (abs(s.capacidades[i] - s.capacidades[i - 1]) <= 1) continue;

				custoParcial = custoOriginal - fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)]
					+ (fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)]
					+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j], fs.n)]);

				if (custoParcial < menorCusto) {
					indiceTrocaI = i;
					indiceTrocaJ = j;
					menorCusto = custoParcial;
				}
			}
		}
	}
	if (indiceTrocaI != -1) {
		return autalizacaoParaSplit(s, fs, indiceTrocaI, indiceTrocaJ, menorCusto);
	} else {
		return s;
	}
}

Solucao perturbar (Solucao s, FabricaSolucao fs) {
	int i = rand() % 2;
	if (i == 1) return splitP(s, fs);
	else return doubleBridge(s, fs);
}