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

Solucao exchange_2_2(Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = s.custo, menorCustoParcial, custoAux = 0,	custoAuxAntigo = 0;
	int indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1;
	short qSumAuxiliar;

	for (int i = 1; i < s.tamanhoCaminho - 2; i++) {
		for (int j = i + 2; j < s.tamanhoCaminho - 2; j++) {

			if (s.ads[0][i - 1].lMin > 0 || s.ads[0][i - 1].lMax < 0) continue;

			if (s.ads[0][i - 1].qSum >= s.ads[j][j + 1].lMin && s.ads[0][i - 1].qSum <= s.ads[j][j + 1].lMax) {
				qSumAuxiliar = s.ads[0][i - 1].qSum + s.ads[j][j + 1].qSum;
				if ((j - 1) - (i + 2) >= 0) {
					int a = i + 2, b = j - 1;
					if (qSumAuxiliar >= s.ads[a][b].lMin && qSumAuxiliar <= s.ads[a][b].lMax)
						qSumAuxiliar += s.ads[a][b].qSum;
					else 
						continue;
				}
				if (qSumAuxiliar >= s.ads[i][i + 1].lMin && qSumAuxiliar <= s.ads[i][i + 1].lMax) {
					qSumAuxiliar += s.ads[i][i + 1].qSum;
					if (qSumAuxiliar >= s.ads[j + 2][indiceFinal].lMin 
						&& qSumAuxiliar <= s.ads[j + 2][indiceFinal].lMax) {

						if (j - (i + 1) > 1) {
							
							custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[i + 2], fs.n)];
							
							custoAux += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[i + 1], s.caminho[j + 2], fs.n)];

							custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[i + 1], s.caminho[i + 2], fs.n)];
							
							custoAuxAntigo += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[j + 2], fs.n)];
						
						} else {
							
							custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[i], fs.n)]
								+ fs.custoArestas[IndiceArestas(s.caminho[i + 1], s.caminho[j + 2], fs.n)];
							
							custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[j + 2], fs.n)]
								+ fs.custoArestas[IndiceArestas(s.caminho[i + 1], s.caminho[j], fs.n)];
						}

						menorCustoParcial = custoOriginal + custoAux - custoAuxAntigo;
						if (menorCustoParcial < menorCusto) {
							indiceTrocaI = i;
							indiceTrocaJ = j;
							menorCusto = menorCustoParcial;
						} 
					}
				}
			}
		}
	}

	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		
		int aux = nova.caminho[indiceTrocaI];
		nova.caminho[indiceTrocaI] = nova.caminho[indiceTrocaJ];
		nova.caminho[indiceTrocaJ] = aux;

		aux = nova.caminho[indiceTrocaI + 1];
		nova.caminho[indiceTrocaI + 1] = nova.caminho[indiceTrocaJ + 1];
		nova.caminho[indiceTrocaJ + 1] = aux;

		nova.custo = menorCusto;

		int capsAntesJ = nova.capacidades[indiceTrocaJ] - nova.capacidades[indiceTrocaJ - 1],
			capsJ_J_1 = nova.capacidades[indiceTrocaJ + 1] - nova.capacidades[indiceTrocaJ];

		int capsAntesI = nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaI - 1],
			capsI_I_1 = nova.capacidades[indiceTrocaI + 1] - nova.capacidades[indiceTrocaI];

		nova.capacidades[indiceTrocaI] = nova.capacidades[indiceTrocaI - 1] + capsAntesJ;

		for (int a = indiceTrocaI + 1; a < indiceTrocaJ; a++) {
			aux = capsJ_J_1;
			capsJ_J_1 = nova.capacidades[a + 1] - nova.capacidades[a];
			nova.capacidades[a] = nova.capacidades[a - 1] + aux;
		}

		nova.capacidades[indiceTrocaJ] = nova.capacidades[indiceTrocaJ - 1] + capsAntesI;
		nova.capacidades[indiceTrocaJ + 1] = nova.capacidades[indiceTrocaJ] + capsI_I_1;

		merge(&nova, fs.q, indiceTrocaI, indiceTrocaJ + 1);

		return nova;
	} else {
		return s;
	}
}

Solucao exchange_1_2(Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = s.custo, menorCustoParcial, custoAux = 0,	custoAuxAntigo = 0;
	int indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1;
	int iniSeg2, iniSeg3, iniSeg4, iniSeg5, fimSeg1, fimSeg2, fimSeg3, fimSeg4;
	short qSumAuxiliar;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		for (int j = 1; j < s.tamanhoCaminho - 2; j++) {

			if (i == j) continue;
			if (i > j && i - j < 2) continue;
			if (s.ads[0][i - 1].lMin > 0 || s.ads[0][i - 1].lMax < 0) continue;

			if (i < j) {
				fimSeg1 = i - 1, iniSeg2 = j, fimSeg2 = j + 1, iniSeg3 = i + 1,
					fimSeg3 = j - 1, iniSeg4 = i, fimSeg4 = i, iniSeg5 = j + 2;
			} else {
				fimSeg1 = j - 1, iniSeg2 = i, fimSeg2 = i, iniSeg3 = j + 2,
					fimSeg3 = i - 1, iniSeg4 = j, fimSeg4 = j + 1, iniSeg5 = i + 1;
			}

			if (s.ads[0][fimSeg1].qSum >= s.ads[iniSeg2][fimSeg2].lMin && s.ads[0][fimSeg1].qSum <= s.ads[iniSeg2][fimSeg2].lMax) {
			
				qSumAuxiliar = s.ads[0][fimSeg1].qSum + s.ads[iniSeg2][fimSeg2].qSum;
				
				if (qSumAuxiliar >= s.ads[iniSeg3][fimSeg3].lMin && qSumAuxiliar <= s.ads[iniSeg3][fimSeg3].lMax) {
				
					qSumAuxiliar += s.ads[iniSeg3][fimSeg3].qSum;

					if (qSumAuxiliar >= s.ads[iniSeg4][fimSeg4].lMin && qSumAuxiliar <= s.ads[iniSeg4][fimSeg4].lMax) {

						qSumAuxiliar += s.ads[iniSeg4][fimSeg4].qSum;
						if (qSumAuxiliar >= s.ads[iniSeg5][indiceFinal].lMin && qSumAuxiliar <= s.ads[iniSeg5][indiceFinal].lMax) {

							
							if (i + 1 != j && j + 2 != i) {
								
								custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
									+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[i + 1], fs.n)];
								
								custoAux += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)] 
									+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j + 2], fs.n)];

								custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
									+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
								
								custoAuxAntigo += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)] 
									+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[j + 2], fs.n)];
							
							} else {
								
								custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
									+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[i], fs.n)]
									+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j + 2], fs.n)];
								
								custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
									+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j], fs.n)]
									+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[j + 2], fs.n)];
							}

							menorCustoParcial = custoOriginal + custoAux - custoAuxAntigo;
							if (menorCustoParcial < menorCusto) {
								indiceTrocaI = i;
								indiceTrocaJ = j;
								menorCusto = menorCustoParcial;
							} 

						}
					}
				}
			}
		}
	}
	
	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		nova.custo = menorCusto;

		int auxI = nova.caminho[indiceTrocaI], auxJ = nova.caminho[indiceTrocaJ],
			capsAntesI = nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaI - 1],
			capsAntesJ = nova.capacidades[indiceTrocaJ] - nova.capacidades[indiceTrocaJ - 1],
			capsJ_J_1 = nova.capacidades[indiceTrocaJ + 1] - nova.capacidades[indiceTrocaJ];

		if (indiceTrocaI < indiceTrocaJ) {
			memmove(nova.caminho + indiceTrocaI + 2, nova.caminho + indiceTrocaI + 1, sizeof(int) * (indiceTrocaJ - indiceTrocaI - 1));
			nova.caminho[indiceTrocaI] = auxJ;
			nova.caminho[indiceTrocaI + 1] = nova.caminho[indiceTrocaJ + 1];
			nova.caminho[indiceTrocaJ + 1] = auxI;

			int capsDpsI = nova.capacidades[indiceTrocaI + 1] - nova.capacidades[indiceTrocaI];

			memmove(nova.capacidades + indiceTrocaI + 2, nova.capacidades + indiceTrocaI + 1, sizeof(int) * (indiceTrocaJ - indiceTrocaI - 1));
			nova.capacidades[indiceTrocaI] = nova.capacidades[indiceTrocaI - 1] + capsAntesJ;
			nova.capacidades[indiceTrocaI + 1] = nova.capacidades[indiceTrocaI] + capsJ_J_1;

			for (int a = indiceTrocaI + 2; a < indiceTrocaJ + 1; a++) {
				auxJ = capsDpsI;
				capsDpsI = nova.capacidades[a + 1] - nova.capacidades[a];
				nova.capacidades[a] = nova.capacidades[a - 1] + auxJ;
			}

			nova.capacidades[indiceTrocaJ + 1] = nova.capacidades[indiceTrocaJ] + capsAntesI;

			merge(&nova, fs.q, indiceTrocaI, indiceTrocaJ);

		} else {
			nova.caminho[indiceTrocaJ] = auxI;
			nova.caminho[indiceTrocaI] = nova.caminho[indiceTrocaJ + 1];
			memcpy(nova.caminho + indiceTrocaJ + 1, nova.caminho + indiceTrocaJ + 2, sizeof(int) * (indiceTrocaI - (indiceTrocaJ + 1) - 1));
			nova.caminho[indiceTrocaI - 1] = auxJ;

			int capsDpsJ = nova.capacidades[indiceTrocaJ + 2] - nova.capacidades[indiceTrocaJ];

			nova.capacidades[indiceTrocaJ] = nova.capacidades[indiceTrocaJ - 1] + capsAntesI;

			for (int a = indiceTrocaJ + 1; a < indiceTrocaI - 1; a++) {
				nova.capacidades[a] = nova.capacidades[a - 1] + (nova.capacidades[a + 1] - nova.capacidades[a]);
			}

			nova.capacidades[indiceTrocaI - 1] = nova.capacidades[indiceTrocaI - 2] + capsAntesJ;
			nova.capacidades[indiceTrocaI] = nova.capacidades[indiceTrocaI - 1] + capsJ_J_1;

			merge(&nova, fs.q, indiceTrocaJ, indiceTrocaI);
		}		

		return nova;
	} else {
		return s;
	}
}

Solucao orOPT(Solucao s, FabricaSolucao fs, ListaTabu lista, int tipo) {

	float menorCusto = INFINITY, custoOriginal = s.custo, menorCustoParcial, menorCustoFinal;
	int indiceTrocaI = -1, indiceTrocaJ = -1, passo, condicaoParada, fimSeg4 = s.tamanhoCaminho - 1,
		fimSeg1, iniSeg2, fimSeg2, iniSeg3, fimSeg3, iniSeg4;
	short qSumAuxiliar;

	if (tipo == 0) { // reinsercao
		passo = 0;
		condicaoParada = s.tamanhoCaminho - 2;
	} else if (tipo == 1) { // orOPT2
	 	passo = 1;
	 	condicaoParada = s.tamanhoCaminho - 3;
	} else if (tipo == 2) { // orOPT3
	 	passo = 2;
	 	condicaoParada = s.tamanhoCaminho - 4;
	} else { // orOPT4
		passo = 3;
		condicaoParada = s.tamanhoCaminho - 5;
	}

	for (int i = 1; i < condicaoParada; i++) {

		menorCustoParcial = custoOriginal - (fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)]
					+ fs.custoArestas[IndiceArestas(s.caminho[i + passo], s.caminho[i + passo + 1], fs.n)]);

		menorCustoParcial += fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i + passo + 1], fs.n)];

		for (int j = 1; j < s.tamanhoCaminho - 1; j++) { // j < s.tamanhoCaminho - 1
			if (i == j) continue;
			if (i < j && j - i < passo + 1) continue; // deve haver uma subsequência de tamanho >= passo + 1 // i == j : j += passo + 1
			if (i > j && i - j < 2) continue; // para os casos em q i está na frente de j
			
			if (i < j && (tabuContem(lista, s.caminho[i - 1], s.caminho[i + passo + 1], i - 1) || tabuContem(lista, s.caminho[j], s.caminho[i], j - (passo + 1)))) continue;
			if (i < j && tabuContem(lista, s.caminho[j + 1], s.caminho[i + passo], j)) continue;
			if (i > j && (tabuContem(lista, s.caminho[j], s.caminho[i], j) || tabuContem(lista, s.caminho[i + passo], s.caminho[j + 1], j + passo + 1))) continue;
			if (i > j && tabuContem(lista, s.caminho[i - 1], s.caminho[i + passo + 1], i + passo)) continue;

			if (i < j) {
				fimSeg1 = i - 1, iniSeg2 = i + passo + 1, fimSeg2 = j, iniSeg3 = i,
					fimSeg3 = i + passo, iniSeg4 = j + 1;
			} else {
				fimSeg1 = j, iniSeg2 = i, fimSeg2 = i + passo, iniSeg3 = j + 1,
					fimSeg3 = i - 1, iniSeg4 = i + passo + 1;
			}

			if (s.ads[0][fimSeg1].lMin > 0 || s.ads[0][fimSeg1].lMax < 0) continue;

			if (s.ads[0][fimSeg1].qSum >= s.ads[iniSeg2][fimSeg2].lMin && s.ads[0][fimSeg1].qSum <= s.ads[iniSeg2][fimSeg2].lMax) {
			
				qSumAuxiliar = s.ads[0][fimSeg1].qSum + s.ads[iniSeg2][fimSeg2].qSum;
				
				if (qSumAuxiliar >= s.ads[iniSeg3][fimSeg3].lMin && qSumAuxiliar <= s.ads[iniSeg3][fimSeg3].lMax) {
				
					qSumAuxiliar += s.ads[iniSeg3][fimSeg3].qSum;

					if (qSumAuxiliar >= s.ads[iniSeg4][fimSeg4].lMin && qSumAuxiliar <= s.ads[iniSeg4][fimSeg4].lMax) {
						menorCustoFinal = menorCustoParcial + fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[i], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[i + passo], s.caminho[j + 1], fs.n)];

						menorCustoFinal -= fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[j + 1], fs.n)];
						if (menorCustoFinal < menorCusto) {
							menorCusto = menorCustoFinal;
							indiceTrocaI = i;
							indiceTrocaJ = j;
						}
					}
				}
			}
		}
	}
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

Solucao reinsercao(Solucao s, FabricaSolucao fs, ListaTabu lista) {
	return orOPT(s, fs, lista, 0);
}

Solucao orOPT2(Solucao s, FabricaSolucao fs, ListaTabu lista) {
	return orOPT(s, fs, lista, 1);
}

Solucao orOPT3(Solucao s, FabricaSolucao fs, ListaTabu lista) {
	return orOPT(s, fs, lista, 2);
}

Solucao orOPT4(Solucao s, FabricaSolucao fs, ListaTabu lista) {
	return orOPT(s, fs, lista, 3);
}

Solucao _2OPT (Solucao s, FabricaSolucao fs, ListaTabu lista) {
	float menorCusto = INFINITY, custoOriginal = s.custo, custoParcial;
	int aux, indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1, auxI, auxJ;

	short qSomaI, lMinI, lMaxI, qSomaAuxiliar;
	ADS ads;

	for (int i = 0; i < s.tamanhoCaminho; i++) {
		for (int j = i + 3; j < s.tamanhoCaminho; j++) {

			if (s.ads[0][i].lMin > 0 || s.ads[0][i].lMax < 0) continue;
			if (tabuContem(lista, s.caminho[i], s.caminho[j-1], i) || tabuContem(lista, s.caminho[i+1], s.caminho[j], j)) continue;

			auxI = i + 1, auxJ = j - 1;
			ads = s.ads[auxI][auxJ];
			qSomaI = ads.qSum;
			lMinI = -ads.qSum + ads.qMax;
			lMaxI = fs.q - ads.qSum + ads.qMin;

			if (s.ads[0][i].qSum >= lMinI && s.ads[0][i].qSum <= lMaxI) {
				qSomaAuxiliar = s.ads[0][i].qSum + qSomaI;
				if (qSomaAuxiliar >= s.ads[j][indiceFinal].lMin && qSomaAuxiliar <= s.ads[j][indiceFinal].lMax) {

					custoParcial = custoOriginal - (
						fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[auxI], fs.n)]
							+ fs.custoArestas[IndiceArestas(s.caminho[auxJ], s.caminho[j], fs.n)])
						+ (fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[auxJ], fs.n)]
							+ fs.custoArestas[IndiceArestas(s.caminho[auxI], s.caminho[j], fs.n)]);

					if (custoParcial < menorCusto) {
						menorCusto = custoParcial;
						indiceTrocaI = i;
						indiceTrocaJ = j;
					}
				}
			}
		}
	}

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

Solucao atualizacaoParaSplit (Solucao s, FabricaSolucao fs, int indiceTrocaI, int indiceTrocaJ, short qSumNovaVisita, float novoCusto) {
	Solucao copia = copiarSolucao(s);
	copia.tamanhoCaminho += 1;
	copia.caminho = (int *) realloc(copia.caminho, sizeof(int) * copia.tamanhoCaminho);
	copia.capacidades = (int *) realloc(copia.capacidades, sizeof(int) * copia.tamanhoCaminho);
	copia.custo = novoCusto;

	int aux = copia.caminho[indiceTrocaI];
	memcpy(copia.caminho + indiceTrocaJ + 1, copia.caminho + indiceTrocaJ, sizeof(int) * (s.tamanhoCaminho - indiceTrocaJ));
	copia.caminho[indiceTrocaJ] = aux;

	int inicioLoop, fimLoop, fatorAlteracao, fatorAlteracaoCapacidade;
	short qSumDivididoI = copia.ads[indiceTrocaI][indiceTrocaI].qSum - qSumNovaVisita;

	int diff = copia.capacidades[indiceTrocaI] - copia.capacidades[indiceTrocaI - 1], diffAbs = abs(diff);

	if (indiceTrocaI < indiceTrocaJ) {
		fatorAlteracao = diff < 0 ? 1 : -1;
		fatorAlteracaoCapacidade = fatorAlteracao * abs(abs(qSumDivididoI) - diffAbs);
		inicioLoop = indiceTrocaI, fimLoop = indiceTrocaJ;
	} else {
		inicioLoop = indiceTrocaJ, fimLoop = indiceTrocaI + 1;
		fatorAlteracao = diff < 0 ? -1 : 1;
		fatorAlteracaoCapacidade = fatorAlteracao * abs(abs(qSumDivididoI) - diffAbs);
	}

	aux = copia.capacidades[indiceTrocaI];
	memcpy(copia.capacidades + indiceTrocaJ + 1, copia.capacidades + indiceTrocaJ, sizeof(int) * (s.tamanhoCaminho - indiceTrocaJ));
	copia.capacidades[indiceTrocaJ] = aux;

	for (int a = inicioLoop; a < fimLoop; a++) copia.capacidades[a] += fatorAlteracaoCapacidade;

	if (indiceTrocaI < indiceTrocaJ)
		fatorAlteracao *= -1;
	copia.capacidades[indiceTrocaJ] = copia.capacidades[indiceTrocaJ - 1] + fatorAlteracao * abs(qSumNovaVisita);

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

Solucao split (Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = s.custo, custoParcial;
	short qSum, lMin2, lMax2, qSum2, lMin4, lMax4, qSum4, qSumNovaVisita, melhorQSumNovaVisita;
	int fimSeg1, iniSeg3, fimSeg3, iniSeg5, indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		if (fs.demandas[s.caminho[i]] < - 1 || fs.demandas[s.caminho[i]] > 1) {
			for (int j = 1; j < s.tamanhoCaminho - 1; j++) {

				if (i == j) continue;
				if (s.caminho[i] == s.caminho[j] || s.caminho[i] == s.caminho[j - 1] || s.caminho[j] == s.caminho[i - 1]) continue;
				if (abs(s.capacidades[i] - s.capacidades[i - 1]) <= 1) continue;

				if (fs.demandas[s.caminho[i]] < - 1) { // coleta
					
					qSum = s.ads[i][i].qSum % 2 == 0 ? s.ads[i][i].qSum / 2 : s.ads[i][i].qSum / 2 + 1;
					qSumNovaVisita = s.ads[i][i].qSum - qSum;

					if (i < j) {
						lMin4 = 0, lMax4 = fs.q - qSumNovaVisita, qSum4 = qSumNovaVisita,
							qSum2 = qSum, lMin2 = 0, lMax2 = fs.q - qSum;
					} else {
						lMin2 = 0, lMax2 = fs.q - qSumNovaVisita, qSum2 = qSumNovaVisita,
							qSum4 = qSum, lMin4 = 0, lMax4 = fs.q - qSum;
					}
				} else { // entrega

					qSum = s.ads[i][i].qSum % 2 == 0 ? s.ads[i][i].qSum / 2 : s.ads[i][i].qSum / 2 - 1;
					qSumNovaVisita = s.ads[i][i].qSum - qSum;

					if (i < j) {
						lMin4 = -qSumNovaVisita, lMax4 = fs.q, qSum4 = qSumNovaVisita,
							qSum2 = qSum, lMin2 = -qSum, lMax2 = fs.q;
					} else {
						lMin2 = -qSumNovaVisita, lMax2 = fs.q, qSum2 = qSumNovaVisita,
							qSum4 = qSum, lMin4 = -qSum, lMax4 = fs.q;
					}
				}

				if (i < j) {
					fimSeg1 = i - 1, iniSeg3 = i + 1, fimSeg3 = j - 1, iniSeg5 = j;
				} else {
					fimSeg1 = j - 1, iniSeg3 = j, fimSeg3 = i - 1, iniSeg5 = i + 1;
				}

				if (s.ads[0][fimSeg1].lMin > 0 || s.ads[0][fimSeg1].lMax < 0) continue;

				if (s.ads[0][fimSeg1].qSum >= lMin2 && s.ads[0][fimSeg1].qSum <= lMax2) {
					qSum = s.ads[0][fimSeg1].qSum + qSum2;
					
					if (qSum >= s.ads[iniSeg3][fimSeg3].lMin && qSum <= s.ads[iniSeg3][fimSeg3].lMax) {
						qSum += s.ads[iniSeg3][fimSeg3].qSum;
						
						if (qSum >= lMin4 && qSum <= lMax4) {
							qSum += qSum4;
							if (qSum >= s.ads[iniSeg5][indiceFinal].lMin && qSum <= s.ads[iniSeg5][indiceFinal].lMax) {

								custoParcial = custoOriginal - fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)]
									+ (fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)]
									+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j], fs.n)]);

								if (custoParcial < menorCusto) {
									indiceTrocaI = i;
									indiceTrocaJ = j;
									menorCusto = custoParcial;
									melhorQSumNovaVisita = qSumNovaVisita;
								}
							}
						}
					}
				}
			}
		}
	}
	if (indiceTrocaI != -1) {
		return atualizacaoParaSplit(s, fs, indiceTrocaI, indiceTrocaJ, melhorQSumNovaVisita, menorCusto);
	} else {
		return s;
	}
}

/*Solucao RVND (Solucao s, FabricaSolucao fs, ListaTabu lista) {
	Solucao melhorSolucao = copiarSolucao(s), sLinha;
	int indices[] = {0, 1, 2, 3, 4, 5, 6};
	Solucao (*vizinhancas[])(Solucao, FabricaSolucao, ListaTabu) = {split, reinsercao, _2OPT, orOPT2, orOPT3, orOPT4, exchange_2_2};
	int LN = 7, N, aux;
	float melhorCusto = INFINITY;

	while (LN > 0) {
		N = rand() % LN;
		sLinha = (*vizinhancas[indices[N]])(melhorSolucao, fs, lista);
		if (sLinha.custo < melhorCusto) {
			if (sLinha.caminho != melhorSolucao.caminho) liberarSolucao(melhorSolucao);
			melhorSolucao = sLinha;
			melhorCusto = sLinha.custo;
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
	return melhorSolucao;
}*/

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
		return s;//autalizacaoParaSplit(s, fs, indiceTrocaI, indiceTrocaJ, menorCusto);
	} else {
		return s;
	}
}

Solucao perturbar (Solucao s, FabricaSolucao fs) {
	int i = rand() % 2;
	if (i == 1) return splitP(s, fs);
	else return doubleBridge(s, fs);
}
