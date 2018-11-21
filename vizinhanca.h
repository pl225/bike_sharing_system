Solucao swap(Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = s.custo, menorCustoParcial, custoAux = 0,	custoAuxAntigo = 0;
	int indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1;
	short qSumAuxiliar;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		for (int j = i + 1; j < s.tamanhoCaminho - 1; j++) {

			if (s.ads[0][i].lMin > 0 || s.ads[0][i].lMax < 0) continue;

			if (s.ads[0][i - 1].qSum >= s.ads[j][j].lMin && s.ads[0][i - 1].qSum <= s.ads[j][j].lMax) {
				qSumAuxiliar = s.ads[0][i - 1].qSum + s.ads[j][j].qSum;
				if ((j - 1) - (i + 1) > 0) {
					int a = i + 1, b = j - 1;
					if (qSumAuxiliar >= s.ads[a][b].lMin && qSumAuxiliar <= s.ads[a][b].lMax)
						qSumAuxiliar += s.ads[a][b].qSum;
					else 
						continue;
				}
				if (qSumAuxiliar >= s.ads[i][i].lMin && qSumAuxiliar <= s.ads[i][i].lMax) {
					qSumAuxiliar += s.ads[i][i].qSum;
					if (qSumAuxiliar >= s.ads[j + 1][indiceFinal].lMin 
						&& qSumAuxiliar <= s.ads[j + 1][indiceFinal].lMax) {

						if (j - i > 1) {
							custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[i + 1], fs.n)];
							custoAux += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j + 1], fs.n)];

							custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
							custoAuxAntigo += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[j + 1], fs.n)];
						} else {
							custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j + 1], fs.n)];
							custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
								+ fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[j + 1], fs.n)];
						}

						menorCustoParcial = custoOriginal + custoAux - custoAuxAntigo;
						if (menorCustoParcial <= menorCusto) {
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
			atualizarADS(nova, fs.q, indiceTrocaI, indiceTrocaJ);
		}

		return nova;
	} else {
		return s;
	}
}

Solucao orOPT(Solucao s, FabricaSolucao fs, int tipo) {

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

		for (int j = 1; j < condicaoParada; j++) {
			if (i == j) continue;
			if (i < j && j - i < passo + 1) continue; // deve haver uma subsequência de tamanho >= passo + 1 // i == j : j += passo + 1
			if (i > j && i - j < 2) continue; // para os casos em q i está na frente de j

			if (i < j) {
				fimSeg1 = i - 1, iniSeg2 = i + passo + 1, fimSeg2 = j, iniSeg3 = i,
					fimSeg3 = i + passo, iniSeg4 = j + 1;
			} else {
				fimSeg1 = j, iniSeg2 = i, fimSeg2 = i + passo, iniSeg3 = j + 1,
					fimSeg3 = i - 1, iniSeg4 = i + passo + 1;
			}

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
			atualizarADS(nova, fs.q, indiceTrocaI, indiceTrocaJ);
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

			atualizarADS(nova, fs.q, indiceTrocaJ + 1, indiceTrocaI + passo);	
		}
	
		return nova;
	} else {
		return s;
	}
}

Solucao reinsercao(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 0);
}

Solucao orOPT2(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 1);
}

Solucao orOPT3(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 2);
}

Solucao orOPT4(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 3);
}

Solucao _2OPT (Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = s.custo, custoParcial;
	int aux, indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1, auxI, auxJ;

	short qSomaI, qMinI, qMaxI, lMinI, lMaxI, qSomaAuxiliar;
	ADS ads;

	for (int i = 0; i < s.tamanhoCaminho; i++) {
		for (int j = i + 3; j < s.tamanhoCaminho; j++) {
			auxI = i + 1, auxJ = j - 1;
			ads = s.ads[auxI][auxJ];
			qSomaI = ads.qSum;
			qMinI = ads.qSum - ads.qMax;
			qMaxI = ads.qSum - ads.qMin;
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
		atualizarADS(copia, fs.q, indiceTrocaI + 1, indiceTrocaJ - 1);		
		return copia;
	} else {
		return s;
	}
}

Solucao split (Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = custo(s, fs), custoParcial;
	short qSum, lMin2, lMax2, qSum2, lMin4, lMax4, qSum4;
	int fimSeg1, iniSeg3, fimSeg3, iniSeg5, indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		if (fs.demandas[s.caminho[i]] < - 1 || fs.demandas[s.caminho[i]] > 1) {
			for (int j = 1; j < s.tamanhoCaminho - 1; j++) {

				if (i == j) continue;
				if (s.caminho[i] == s.caminho[j] || s.caminho[i] == s.caminho[j - 1] || s.caminho[i] == s.caminho[j + 1]) continue;

				if (fs.demandas[s.caminho[i]] < - 1) { // coleta
					qSum = s.ads[i][i].qSum - 1;
					if (i < j) {
						lMin4 = 0, lMax4 = fs.q - 1, qSum4 = 1,
							qSum2 = qSum, lMin2 = 0, lMax2 = fs.q - qSum;
					} else {
						lMin2 = 0, lMax2 = fs.q - 1, qSum2 = 1,
							qSum4 = qSum, lMin4 = 0, lMax4 = fs.q - qSum;
					}
				} else { // entrega
					qSum = s.ads[i][i].qSum + 1;
					if (i < j) {
						lMin4 = 1, lMax4 = fs.q, qSum4 = -1,
							qSum2 = qSum, lMin2 = -qSum, lMax2 = fs.q;
					} else {
						lMin2 = 1, lMax2 = fs.q, qSum2 = -1,
							qSum4 = qSum, lMin4 = -qSum, lMax4 = fs.q;
					}
				}

				if (i < j) {
					fimSeg1 = i - 1, iniSeg3 = i + 1, fimSeg3 = j - 1, iniSeg5 = j;
				} else {
					fimSeg1 = j - 1, iniSeg3 = j, fimSeg3 = i - 1, iniSeg5 = i + 1;
				}

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
								}
							}
						}
					}
				}
			}
		}
	}
	if (indiceTrocaI != -1) {
		Solucao copia = copiarSolucao(s);
		copia.tamanhoCaminho += 1;
		copia.caminho = (int *) realloc(copia.caminho, sizeof(int) * copia.tamanhoCaminho);
		copia.capacidades = (int *) realloc(copia.capacidades, sizeof(int) * copia.tamanhoCaminho);

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
		for (int i = 0; i < copia.tamanhoCaminho; i++) {
			if (i != indiceTrocaJ) {
				copia.ads[i] = (ADS*) realloc(copia.ads[i], sizeof(ADS) * copia.tamanhoCaminho);
				memcpy(copia.ads[i] + indiceTrocaJ + 1, copia.ads[i] + indiceTrocaJ, sizeof(ADS) * (s.tamanhoCaminho - indiceTrocaJ));
			}
		}

		atualizarADS(copia, fs.q, inicioLoop, fimLoop);

		return copia;
	} else {
		return s;
	}
}

Solucao RVND (Solucao s, FabricaSolucao fs) {
	Solucao melhorSolucao = s, sLinha;
	int indices[] = {0, 1, 2, 3, 4, 5, 6};
	Solucao (*vizinhancas[])(Solucao, FabricaSolucao) = {split, reinsercao, _2OPT, orOPT2, orOPT3, orOPT4, swap};
	int LN = 7, N, aux;

	while (LN > 0) {
		N = rand() % LN;
		sLinha = (*vizinhancas[N])(melhorSolucao, fs);
		if (sLinha.custo < melhorSolucao.custo) {
			melhorSolucao = sLinha;
			LN = 7;
		} else {
			LN--;
			if (N < 6) {
				aux = indices[N];
				memcpy(indices + N, indices + N + 1, sizeof(int) * (6 - N));
				indices[6] = aux;
			}
		}
	}

	return melhorSolucao;
}

Solucao doubleBridge (Solucao s, FabricaSolucao fs) {
	
}

Solucao splitP (Solucao s, FabricaSolucao fs) {
	float menorCusto = INFINITY, custoOriginal = custo(s, fs), custoParcial;
	short qSum, lMin2, lMax2, qSum2, lMin4, lMax4, qSum4;
	int fimSeg1, iniSeg3, fimSeg3, iniSeg5, indiceTrocaI = -1, indiceTrocaJ = -1, indiceFinal = s.tamanhoCaminho - 1;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		if (fs.demandas[s.caminho[i]] < - 1 || fs.demandas[s.caminho[i]] > 1) {
			for (int j = 1; j < s.tamanhoCaminho - 1; j++) {

				if (i == j) continue;
				if (s.caminho[i] == s.caminho[j] || s.caminho[i] == s.caminho[j - 1] || s.caminho[i] == s.caminho[j + 1]) continue;

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
		Solucao copia = copiarSolucao(s);
		copia.tamanhoCaminho += 1;
		copia.caminho = (int *) realloc(copia.caminho, sizeof(int) * copia.tamanhoCaminho);
		copia.capacidades = (int *) realloc(copia.capacidades, sizeof(int) * copia.tamanhoCaminho);

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
		for (int i = 0; i < copia.tamanhoCaminho; i++) {
			if (i != indiceTrocaJ) {
				copia.ads[i] = (ADS*) realloc(copia.ads[i], sizeof(ADS) * copia.tamanhoCaminho);
				memcpy(copia.ads[i] + indiceTrocaJ + 1, copia.ads[i] + indiceTrocaJ, sizeof(ADS) * (s.tamanhoCaminho - indiceTrocaJ));
			}
		}

		atualizarADS(copia, fs.q, inicioLoop, fimLoop);

		return copia;
	} else {
		return s;
	}
}
