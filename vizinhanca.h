float custo (Solucao s, FabricaSolucao fs) { // mover
	float f = 0;
	for (int i = 0; i < s.tamanhoCaminho - 1; i++) {
		f += fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
	}
	return f;
}

Solucao copiarSolucao (Solucao s) { // mover
	Solucao copia;
	copia.tamanhoCaminho = s.tamanhoCaminho;

	size_t tamanhoInteiroTotal = sizeof(int) * s.tamanhoCaminho;

	copia.caminho = (int *) malloc(tamanhoInteiroTotal);
	copia.capacidades = (int *) malloc(tamanhoInteiroTotal);
	copia.viavel = s.viavel;
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

Solucao swap(Solucao s, FabricaSolucao fs) {
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, menorCustoParcial, custoAux = 0,	custoAuxAntigo = 0;
	int indiceTrocaI = -1, indiceTrocaJ = -1;
	short qSumAuxiliar;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		for (int j = i + 1; j < s.tamanhoCaminho - 1; j++) {

			if (s.ads[0][i - 1].qSum >= s.ads[j][j].lMin && s.ads[0][i - 1].qSum <= s.ads[j][j].lMax) {
				qSumAuxiliar = s.ads[0][i - 1].qSum + s.ads[j][j].qSum;
				if ((j - 1) - (i + 1) > 0) {
					int a = i + 1, b = j - 1;
					if (qSumAuxiliar >= s.ads[a][b].lMin && qSumAuxiliar <= s.ads[a][b].lMax)
						qSumAuxiliar += s.ads[a][b].qSum;
				}
				if (qSumAuxiliar >= s.ads[i][i].lMin && qSumAuxiliar <= s.ads[i][i].lMax) {
					qSumAuxiliar += s.ads[i][i].qSum;
					if (qSumAuxiliar >= s.ads[j + 1][s.tamanhoCaminho - 1].lMin 
						&& qSumAuxiliar <= s.ads[j + 1][s.tamanhoCaminho - 1].lMax) {

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

		int qI = nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaI - 1];
		int qJ = nova.capacidades[indiceTrocaJ] - nova.capacidades[indiceTrocaJ - 1];

		if (qI != qJ) {
			int diff = nova.capacidades[indiceTrocaJ] > nova.capacidades[indiceTrocaI] ?
					nova.capacidades[indiceTrocaJ] - nova.capacidades[indiceTrocaI] : nova.capacidades[indiceTrocaI] - nova.capacidades[indiceTrocaJ];
			for (int i = indiceTrocaI; i < indiceTrocaJ; i++)
				nova.capacidades[i] += diff;
			atualizarADS(nova, fs.q, indiceTrocaI, indiceTrocaJ);
		}

		return nova;
	} else {
		return s;
	}
}

Solucao orOPT(Solucao s, FabricaSolucao fs, int tipo) {

	float menorCusto = custo(s, fs), custoOriginal = menorCusto, menorCustoParcial, menorCustoFinal;
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
			memcpy(nova.capacidades + indiceTrocaI, nova.capacidades + posAux, sizeof(int) * trazidosTras);
			nova.capacidades[inicioTrocaCapacidade] = nova.capacidades[inicioTrocaCapacidade - 1] + diff;
			for (int i = inicioTrocaCapacidade + 1, a = 1; a < passo + 1; i++, a++) {
				nova.capacidades[i] = nova.capacidades[i - 1] + (capacidadesMovidas[a] - capacidadesMovidas[a - 1]);
			} 
			atualizarADS(nova, fs.q, indiceTrocaI, indiceTrocaJ);
		} else {
			int posAux = indiceTrocaJ + passo + 1;
			int trazidosFrente = indiceTrocaI - indiceTrocaJ;
			memcpy(nova.caminho + posAux, nova.caminho + indiceTrocaJ, sizeof(int) * trazidosFrente);
			
			memcpy(nova.caminho + indiceTrocaJ + 1, verticesMovidos, sizeof(int) * (passo + 1));

			int diff = nova.capacidades[indiceTrocaJ + 1] - nova.capacidades[indiceTrocaJ];
			int inicioTrocaCapacidade = posAux + 1;
			nova.capacidades[inicioTrocaCapacidade] = capacidadesMovidas[passo] + diff;

			for (int i = inicioTrocaCapacidade + 1, a = indiceTrocaJ + 2; a < inicioTrocaCapacidade; i++, a++) {
				nova.capacidades[i] = nova.capacidades[i - 1] + (nova.capacidades[a] - nova.capacidades[a - 1]);
			}
			memcpy(nova.capacidades + indiceTrocaJ + 1, capacidadesMovidas, sizeof(int) * (passo + 1));
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
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, custoParcial;
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
		for (int a = indiceTrocaI + 1, b = indiceTrocaJ - 1; a < b; a++, b--) { // desfazendo a reversão entre i e j
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
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, custoParcial;
	int aux, melhorCaminho[s.tamanhoCaminho + 1], melhorCapacidade[s.tamanhoCaminho + 1], viavel = TRUE, demandas[fs.n];
	Solucao copia = copiarSolucao(s);

	copia.tamanhoCaminho += 1;
	copia.caminho = (int *) realloc(copia.caminho, sizeof(int) * copia.tamanhoCaminho);
	copia.capacidades = (int *) realloc(copia.capacidades, sizeof(int) * copia.tamanhoCaminho);
	memcpy(demandas, fs.demandas, sizeof(int) * fs.n);

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		if (fs.demandas[s.caminho[i]] < - 1 || fs.demandas[s.caminho[i]] > 1) {
			for (int j = 1; j < s.tamanhoCaminho - 1; j++) {

				int inicioLoop1, fimLoop1, inicioLoop2, fimLoop2, fatorAlteracao;

				if (i != j) {

					custoParcial = custoOriginal - fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)]
								+ (fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)]
									+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j], fs.n)]);
					
					aux = copia.caminho[i];
					memcpy(copia.caminho + j + 1, copia.caminho + j, sizeof(int) * (s.tamanhoCaminho - j));
					copia.caminho[j] = aux;

					aux = copia.capacidades[i];
					memcpy(copia.capacidades + j + 1, copia.capacidades + j, sizeof(int) * (s.tamanhoCaminho - j));
					copia.capacidades[j] = aux;

					if (i < j) {
						inicioLoop1 = i, fimLoop1 = j;
						inicioLoop2 = j + 1, fimLoop2 = copia.tamanhoCaminho;
					} else {
						inicioLoop1 = i, fimLoop1 = copia.tamanhoCaminho;
						inicioLoop2 = j, fimLoop2 = i;
					}

					fatorAlteracao = copia.capacidades[i] - copia.capacidades[i - 1] < 0 ? 1 : -1;
					for (int a = inicioLoop1; a < fimLoop1; a++) copia.capacidades[a] += fatorAlteracao;

					fatorAlteracao *= -1;
					copia.capacidades[j] = copia.capacidades[j - 1] + fatorAlteracao;
					//for (int a = inicioLoop2; a < fimLoop2; a++) copia.capacidades[a] += fatorAlteracao;
					
					for (int a = 1; a < copia.tamanhoCaminho - 1; a++) {
						if (copia.capacidades[a] < 0 || copia.capacidades[a] > fs.q) {
							viavel = FALSE;
							break;
						}
						demandas[copia.caminho[a]] += (-1) * (copia.capacidades[a] - copia.capacidades[a - 1]);
					}

					if (viavel) {
						for (int a = 0; a < fs.n; a++) {
							if (demandas[a] != 0) {
								viavel = FALSE;
								break;
							}
						}
					}

					if ((copia.viavel == TRUE && viavel == TRUE && custoParcial < menorCusto) || 
						(copia.viavel == FALSE && viavel == TRUE)) {

						copia.viavel = viavel;	
						menorCusto = custoParcial;
						memcpy(melhorCaminho, copia.caminho, sizeof(int) * copia.tamanhoCaminho);
						memcpy(melhorCapacidade, copia.capacidades, sizeof(int) * copia.tamanhoCaminho);
					}

					memcpy(copia.caminho + j, copia.caminho + j + 1, sizeof(int) * (s.tamanhoCaminho - j));
					memcpy(copia.capacidades, s.capacidades, sizeof(int) * s.tamanhoCaminho);
					memcpy(demandas, fs.demandas, sizeof(int) * fs.n);
					viavel = TRUE;
				}
			}
		}
	}
	
	if (menorCusto < custoOriginal) {
		memcpy(copia.caminho, melhorCaminho, sizeof(int) * copia.tamanhoCaminho);
		memcpy(copia.capacidades, melhorCapacidade, sizeof(int) * copia.tamanhoCaminho);
		return copia;
	} else {
		return s;
	}
}