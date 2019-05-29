#include "vizinhanca.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

unsigned int rand_interval(unsigned int min, unsigned int max)
{
    int r;
    const unsigned int range = 1 + max - min;
    const unsigned int buckets = RAND_MAX / range;
    const unsigned int limit = buckets * range;

    do {
        r = rand();
    } while (r >= limit);

    return min + (r / buckets);
}

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
			if (i < j && j - i < 2) continue;
			if (s.ads[0][i - 1].lMin > 0 || s.ads[0][i - 1].lMax < 0) continue;

			if (i < j) {
				fimSeg1 = i - 1, iniSeg2 = j, fimSeg2 = j + 1, iniSeg3 = i + 1,
					fimSeg3 = j - 1, iniSeg4 = i, fimSeg4 = i, iniSeg5 = j + 2;
			} else {
				fimSeg1 = j - 1, iniSeg2 = i, fimSeg2 = i;
				if (i - j > 2) {
					iniSeg3 = j + 2, fimSeg3 = i - 1, iniSeg4 = j, 
					fimSeg4 = j + 1, iniSeg5 = i + 1;
				} else {
					iniSeg3 = j, fimSeg3 = j + 1, iniSeg4 = i + 1, fimSeg4 = indiceFinal;
					iniSeg5 = -1;
				}
			}

			if (s.ads[0][fimSeg1].qSum >= s.ads[iniSeg2][fimSeg2].lMin && s.ads[0][fimSeg1].qSum <= s.ads[iniSeg2][fimSeg2].lMax) {
			
				qSumAuxiliar = s.ads[0][fimSeg1].qSum + s.ads[iniSeg2][fimSeg2].qSum;
				if (qSumAuxiliar >= s.ads[iniSeg3][fimSeg3].lMin && qSumAuxiliar <= s.ads[iniSeg3][fimSeg3].lMax) {
				
					qSumAuxiliar += s.ads[iniSeg3][fimSeg3].qSum;
					if (qSumAuxiliar >= s.ads[iniSeg4][fimSeg4].lMin && qSumAuxiliar <= s.ads[iniSeg4][fimSeg4].lMax) {

						qSumAuxiliar += s.ads[iniSeg4][fimSeg4].qSum;
						if (iniSeg5 == -1 || (qSumAuxiliar >= s.ads[iniSeg5][indiceFinal].lMin && qSumAuxiliar <= s.ads[iniSeg5][indiceFinal].lMax)) {

							
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
								if (i < j) {
									custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
										+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[i], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j + 2], fs.n)];
									
									custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
										+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[j + 2], fs.n)];
								} else {
									custoAux = fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)] 
										+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[j + 3], fs.n)];
								
									custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)] 
										+ fs.custoArestas[IndiceArestas(s.caminho[j + 1], s.caminho[i], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
								}
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

			merge(&nova, fs.q, indiceTrocaI, indiceTrocaJ + 1);

		} else {
			nova.caminho[indiceTrocaJ] = auxI;
			nova.caminho[indiceTrocaI] = nova.caminho[indiceTrocaJ + 1];
			memcpy(nova.caminho + indiceTrocaJ + 1, nova.caminho + indiceTrocaJ + 2, sizeof(int) * (indiceTrocaI - (indiceTrocaJ + 1) - 1));
			nova.caminho[indiceTrocaI - 1] = auxJ;

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

float custo4Pontos (Solucao s, FabricaSolucao fs, int p1, int p2, int p3, int p4) {
	return s.custo - (fs.custoArestas[IndiceArestas(s.caminho[p1 - 1], s.caminho[p1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p2], s.caminho[p2 + 1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p3 - 1], s.caminho[p3], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p4], s.caminho[p4 + 1], fs.n)]) 

  				   + (fs.custoArestas[IndiceArestas(s.caminho[p1 - 1], s.caminho[p2], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p1], s.caminho[p2 + 1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p3 - 1], s.caminho[p4], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[p3], s.caminho[p4 + 1], fs.n)]);
}

Solucao _3OPT (Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoParcial, custoOriginal = s.custo;
	int indiceTrocaI = -1, indiceTrocaJ = -1, indiceTrocaK = -1, indiceTrocaL = - 1, p2, p4, p6;

	ADS adsA, ads;

	for (int p1 = 1; p1 < s.tamanhoCaminho - 5; p1++) {

		if (s.ads[0][p1].lMin > 0 || s.ads[0][p1].lMax < 0) continue;

		for (int p3 = p1 + 2; p3 < s.tamanhoCaminho - 3; p3++) {

			for (int p5 = p3 + 2; p5 < s.tamanhoCaminho -1; p5++) {

				p2 = p1 + 1, p4 = p3 + 1; p6 = p5 + 1;

				float custoRetirado = fs.custoArestas[IndiceArestas(s.caminho[p1], s.caminho[p2], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p3], s.caminho[p4], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p5], s.caminho[p6], fs.n)];

				if (s.ads[0][p1].qSum >= s.ads[p4][p5].lMin && s.ads[0][p1].qSum <= s.ads[p4][p5].lMax) {

					short qSum = s.ads[0][p1].qSum + s.ads[p4][p5].qSum;
					
					ads = s.ads[p2][p3];
					adsA.qSum = ads.qSum, adsA.lMin = -ads.qSum + ads.qMax, adsA.lMax = fs.q - ads.qSum + ads.qMin;

					if (qSum >= adsA.lMin && qSum <= adsA.lMax) {
						custoParcial = custoOriginal + 
									   (fs.custoArestas[IndiceArestas(s.caminho[p1], s.caminho[p4], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p5], s.caminho[p3], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p2], s.caminho[p6], fs.n)]);
						custoParcial -= custoRetirado;

						if (custoParcial < menorCusto ) {
							menorCusto = custoParcial;
							indiceTrocaI = p4, indiceTrocaJ = p5, indiceTrocaK = p3, indiceTrocaL = p2;
						}
					}

					if (qSum >= ads.lMin && qSum <= ads.lMax) {
						custoParcial = custoOriginal + 
									   (fs.custoArestas[IndiceArestas(s.caminho[p1], s.caminho[p4], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p5], s.caminho[p2], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p3], s.caminho[p6], fs.n)]);
						custoParcial -= custoRetirado;

						if (custoParcial < menorCusto) {
							menorCusto = custoParcial;
							indiceTrocaI = p4, indiceTrocaJ = p5, indiceTrocaK = p2, indiceTrocaL = p3;
						}
					}

				}

				ads = s.ads[p4][p5];
				adsA.qSum = ads.qSum, adsA.lMin = -ads.qSum + ads.qMax, adsA.lMax = fs.q - ads.qSum + ads.qMin;

				if (s.ads[0][p1].qSum >= adsA.lMin && s.ads[0][p1].qSum <= adsA.lMax) {
					
					short qSum = s.ads[0][p1].qSum + adsA.qSum;

					if (qSum >= s.ads[p2][p3].lMin && qSum <= s.ads[p2][p3].lMax) {
						custoParcial = custoOriginal + 
									   (fs.custoArestas[IndiceArestas(s.caminho[p1], s.caminho[p5], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p4], s.caminho[p2], fs.n)]
										+ fs.custoArestas[IndiceArestas(s.caminho[p3], s.caminho[p6], fs.n)]);
						custoParcial -= custoRetirado;

						if (custoParcial < menorCusto) {
							menorCusto = custoParcial;
							indiceTrocaI = p5, indiceTrocaJ = p4, indiceTrocaK = p2, indiceTrocaL = p3;
						}
					}
				}

			}
		}
	}

	if (indiceTrocaI != - 1) {
		Solucao copia = copiarSolucao(s);
		copia.custo = menorCusto;

		int menorIJ = indiceTrocaI < indiceTrocaJ, menorKL = indiceTrocaK < indiceTrocaL;

		if (!menorIJ) inverterSubsequencia(s, copia, indiceTrocaJ, indiceTrocaI);
		if (!menorKL) inverterSubsequencia(s, copia, indiceTrocaL, indiceTrocaK);

		int tamSegIJ = abs(indiceTrocaJ - indiceTrocaI) + 1, tamSegKL = abs(indiceTrocaL - indiceTrocaK) + 1;
		int capKL[tamSegKL], segKL[tamSegKL];

		int menorNumeroKL = menorKL ? indiceTrocaK : indiceTrocaL, menorNumeroIJ = menorIJ ? indiceTrocaI : indiceTrocaJ;

		memcpy(segKL, copia.caminho + menorNumeroKL, sizeof(int) * tamSegKL);

		memcpy(copia.caminho + menorNumeroKL, copia.caminho + menorNumeroIJ, sizeof(int) * tamSegIJ);
		memcpy(copia.caminho + menorNumeroKL + tamSegIJ, segKL, sizeof(int) * tamSegKL);

		memcpy(capKL, copia.capacidades + menorNumeroKL, sizeof(int) * tamSegKL);

		int capInicioKL = copia.capacidades[menorNumeroKL] - copia.capacidades[menorNumeroKL - 1];

		for (int i = menorNumeroKL, j = menorNumeroIJ; i < menorNumeroKL + tamSegIJ; i++, j++)
			copia.capacidades[i] = copia.capacidades[i - 1] + (copia.capacidades[j] - copia.capacidades[j - 1]);

		for (int i = menorNumeroKL + tamSegIJ, a = 0; a < tamSegKL; i++, a++) {
			copia.capacidades[i] = copia.capacidades[i - 1] + capInicioKL;
			capInicioKL = capKL[a + 1] - capKL[a];
		}

		merge(&copia, fs.q, menorNumeroKL, menorIJ ? indiceTrocaJ : indiceTrocaI);		
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

Solucao RVND (Solucao s, FabricaSolucao fs) {
	Solucao melhorSolucao = copiarSolucao(s), sLinha;
	int indices[] = {0, 1, 2, 3};
	Solucao (*vizinhancas[])(Solucao, FabricaSolucao) = {split, _3OPT, exchange_1_2, exchange_2_2};
	int LN = 4, N, aux;
	while (LN > 0) {
		N = rand() % LN;
		sLinha = (*vizinhancas[indices[N]])(melhorSolucao, fs);
		if (sLinha.custo < melhorSolucao.custo) {
			liberarSolucao(melhorSolucao);
			melhorSolucao = sLinha;
			LN = 4;
		} else {
			if (sLinha.caminho != melhorSolucao.caminho) liberarSolucao(sLinha);
			LN--;
			if (N < 3) {
				aux = indices[N];
				memcpy(indices + N, indices + N + 1, sizeof(int) * (3 - N));
				indices[3] = aux;
			}
		}
	}
	if (melhorSolucao.custo == s.custo) {
		liberarSolucao(melhorSolucao);
		return s;
	}
	return melhorSolucao;
}

Solucao _3OPT_P (Solucao s, FabricaSolucao fs) {
	
	Solucao copia = copiarSolucao(s);
	int tamanho = copia.tamanhoCaminho - 1;
	int p1 = rand_interval(1, tamanho - 8),
		p2 = rand_interval(p1 + 1, tamanho - 7),
		p3 = rand_interval(p2 + 2, tamanho - 5),
		p4 = rand_interval(p3 + 1, tamanho - 4),
		p5 = rand_interval(p4 + 2, tamanho - 2),
		p6 = rand_interval(p5 + 1, tamanho - 1);

	copia.custo = s.custo - (fs.custoArestas[IndiceArestas(s.caminho[p1 - 1], s.caminho[p1], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p2], s.caminho[p2 + 1], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p3 - 1], s.caminho[p3], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p4], s.caminho[p4 + 1], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p5 - 1], s.caminho[p5], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p6], s.caminho[p6 + 1], fs.n)]) 
					
						  + (fs.custoArestas[IndiceArestas(s.caminho[p1 - 1], s.caminho[p2], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p1], s.caminho[p2 + 1], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p3 - 1], s.caminho[p4], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p3], s.caminho[p4 + 1], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p5 - 1], s.caminho[p6], fs.n)]
		 						+ fs.custoArestas[IndiceArestas(s.caminho[p5], s.caminho[p6 + 1], fs.n)]);

	inverterSubsequencia(s, copia, p1, p2);
	inverterSubsequencia(s, copia, p3, p4);
	inverterSubsequencia(s, copia, p5, p6);
	
	merge(&copia, fs.q, p1, p6);

	return copia;
}

Solucao splitP (Solucao s, FabricaSolucao fs) {
	int indiceTrocaI = -1, indiceTrocaJ = -1, maior = 1, diff;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		diff = abs(s.capacidades[i] - s.capacidades[i - 1]);
		if (diff > maior) {
			maior = diff;
			indiceTrocaI = i;
		}
	}

	if (indiceTrocaI != -1) {

		indiceTrocaJ = 1 + rand() % (s.tamanhoCaminho - 2);
		while (s.caminho[indiceTrocaJ] == s.caminho[indiceTrocaI] 
			|| s.caminho[indiceTrocaI] == s.caminho[indiceTrocaJ - 1] || s.caminho[indiceTrocaJ] == s.caminho[indiceTrocaI - 1])
			indiceTrocaJ = 1 + rand() % (s.tamanhoCaminho - 2);

		float menorCusto = s.custo - fs.custoArestas[IndiceArestas(s.caminho[indiceTrocaJ - 1], s.caminho[indiceTrocaJ], fs.n)]
				+ (fs.custoArestas[IndiceArestas(s.caminho[indiceTrocaJ - 1], s.caminho[indiceTrocaI], fs.n)]
				+ fs.custoArestas[IndiceArestas(s.caminho[indiceTrocaI], s.caminho[indiceTrocaJ], fs.n)]);

		short qSumNovaVisita, qSum = s.ads[indiceTrocaI][indiceTrocaI].qSum;;

		if (fs.demandas[s.caminho[indiceTrocaI]] < 0)
			qSum = qSum % 2 == 0 ? qSum / 2 : qSum / 2 + 1;
		else
			qSum = qSum % 2 == 0 ? qSum / 2 : qSum / 2 - 1;

		qSumNovaVisita = s.ads[indiceTrocaI][indiceTrocaI].qSum - qSum;

		return atualizacaoParaSplit(s, fs, indiceTrocaI, indiceTrocaJ, qSumNovaVisita, menorCusto);
	} else {
		return s;
	}
}
