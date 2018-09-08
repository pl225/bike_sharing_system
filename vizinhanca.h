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

Solucao reinsercao(Solucao s, FabricaSolucao fs) {
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, custoParcial;
	int aux, melhorCaminho[s.tamanhoCaminho], melhorCapacidade[s.tamanhoCaminho], viavel = TRUE, demandas[fs.n];
	Solucao copia = copiarSolucao(s);
	
	for (int i = 6; i < copia.tamanhoCaminho - 2; i ++) {
		for (int j = 1; j < copia.tamanhoCaminho - 2; j++) {
			if (i != j) {
				custoParcial = custoOriginal - (fs.custoArestas[IndiceArestas(copia.caminho[i - 1], copia.caminho[i], fs.n)]
					+ fs.custoArestas[IndiceArestas(copia.caminho[i], copia.caminho[i + 1], fs.n)]
					+ fs.custoArestas[IndiceArestas(copia.caminho[j], copia.caminho[j + 1], fs.n)])
					+ (fs.custoArestas[IndiceArestas(copia.caminho[i - 1], copia.caminho[i + 1], fs.n)]
					+ fs.custoArestas[IndiceArestas(copia.caminho[j], copia.caminho[i], fs.n)]
					+ fs.custoArestas[IndiceArestas(copia.caminho[i], copia.caminho[j + 1], fs.n)]);

				aux = copia.caminho[i];
				if (i < j) {
					memcpy(copia.caminho + i, copia.caminho + i + 1, sizeof(int) * (j - i));
				} else {
					memcpy(copia.caminho + j + 1, copia.caminho + j, sizeof(int) * (i - j));
				}
				copia.caminho[j] = aux;

				printf("\n");
				for (int a = 0; a < copia.tamanhoCaminho; a++) printf("%d ", copia.caminho[a]);
				printf("\n");

			exit(0);

			}						
		}
	}
}

Solucao orOPT(Solucao s, FabricaSolucao fs, int tipo) {
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, menorCustoParcial;
	int indiceTrocaI = -1, indiceTrocaJ = -1, passo, condicaoParada;

	if (tipo == 0) passo = 2; // orOPT2
	else if (tipo == 1) passo = 3; // orOPT3
	else passo = 4; // orOPT4

	condicaoParada = s.tamanhoCaminho - passo * 2;

	for (int i = 0; i < condicaoParada; i++) {

		if (s.capacidades[i] == s.capacidades[i + passo]) { // verifica se o segmento conserva o fluxo
			
			int diferencaCarga = s.capacidades[i + 1] - s.capacidades[i];

			menorCustoParcial = custoOriginal - (fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[i + passo], s.caminho[i + passo + 1], fs.n)]);

			menorCustoParcial += fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + passo + 1], fs.n)];

			int testeViolacaoCarga;

			for (int j = 0; j < s.tamanhoCaminho - 2; j++) { // verificar condicao de parada
				if (j == i) {
					j += passo + 1;
				}

				testeViolacaoCarga = diferencaCarga < 0 ? 
										s.capacidades[j] + diferencaCarga : s.capacidades[j] + abs(diferencaCarga);

				if (testeViolacaoCarga >=0 && testeViolacaoCarga <= fs.q) {
					menorCustoParcial += fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[i + 1], fs.n)]
						+ fs.custoArestas[IndiceArestas(s.caminho[i + passo], s.caminho[j + 1], fs.n)];

					menorCustoParcial -= fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[j + 1], fs.n)];
					if (menorCustoParcial < menorCusto) {
						menorCusto = menorCustoParcial;
						indiceTrocaI = i;
						indiceTrocaJ = j;
					}
				}
			}
		}
	}
	
	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		int verticesMovidos[passo], capacidadesMovidas[passo];
		
		for (int i = 0, j = indiceTrocaI + 1; i < passo; i++, j++) {
			verticesMovidos[i] = nova.caminho[j];
			capacidadesMovidas[i] = nova.capacidades[j];
		}

		int diferencaNovaCapacidade = nova.capacidades[indiceTrocaJ] - nova.capacidades[indiceTrocaI];

		if (indiceTrocaI < indiceTrocaJ) {
			for (int i = indiceTrocaI + 1; i <= indiceTrocaJ - passo; i++) {
				nova.caminho[i] = nova.caminho[i + passo];
				nova.capacidades[i] = nova.capacidades[i + passo];
			}
			for (int i = indiceTrocaJ - passo + 1, j = 0; j < passo; i++, j++) {
				nova.caminho[i] = verticesMovidos[j];
				nova.capacidades[i] = capacidadesMovidas[j] + diferencaNovaCapacidade;
			}
		} else {
			for (int i = indiceTrocaI + passo; i >= indiceTrocaJ + passo; i--) {
				nova.caminho[i] = nova.caminho[i - passo];
				nova.capacidades[i] = nova.capacidades[i - passo];
			}
			for (int i = indiceTrocaJ + 1, j = 0; j < passo; i++, j++) {
				nova.caminho[i] = verticesMovidos[j];
				nova.capacidades[i] = capacidadesMovidas[j] + diferencaNovaCapacidade;
			}
		}
		return nova;
	} else {
		return s;
	}
}

Solucao orOPT2(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 0);
}

Solucao orOPT3(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 1);
}

Solucao orOPT4(Solucao s, FabricaSolucao fs) {
	return orOPT(s, fs, 2);
}

void custoViabilidade (Solucao s, FabricaSolucao fs, int *viavelRetorno, float *custoRetorno) {
	float custoFinal = 0;
	int q = fs.q, d = 0, demandas[fs.n];
	s.capacidades[0] = q;
	memcpy(demandas, fs.demandas, sizeof(int) * fs.n);

	for (int i = 1; i < s.tamanhoCaminho; i++) {
		custoFinal += fs.custoArestas[IndiceArestas(s.caminho[i-1], s.caminho[i], fs.n)];
		d = demandas[s.caminho[i]];
		if (d != 0) {
			if (d < 0) {
				if (abs(d) <= q) {
				 	q += d;
				 	demandas[s.caminho[i]] = 0;
				} else {
					demandas[s.caminho[i]] += q;
					q = 0;
				}
			}  else {
				if (fs.q - q >= d) {
					q += d;
					demandas[s.caminho[i]] = 0;
				} else {
					demandas[s.caminho[i]] -= fs.q - q;
					q = fs.q;
				}
			}
			s.capacidades[i] = q;
		} else {
			s.capacidades[i] = s.capacidades[i - 1];
		}
	}
	*viavelRetorno = TRUE;	
	for (int i = 0; i < fs.n; i++) {
		if (demandas[i] != 0) {
			*viavelRetorno = FALSE;
			break;
		}
	}
	*custoRetorno = custoFinal;
}

Solucao _2OPT (Solucao s, FabricaSolucao fs) {
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, custoParcial;
	int aux, melhorCaminho[s.tamanhoCaminho], melhorCapacidade[s.tamanhoCaminho], viavel;
	Solucao copia = copiarSolucao(s);

	//memcpy(melhorCaminho, copia.caminho, sizeof(int) * copia.tamanhoCaminho);

	for (int i = 0; i < copia.tamanhoCaminho; i++) { // caminhando pelas combinações
		for (int j = i + 3; j < copia.tamanhoCaminho; j++) {
			
			for (int a = i + 1, b = j - 1; a < b; a++, b--) { // revertendo segmento entre i e j
				aux = copia.caminho[a];
				copia.caminho[a] = copia.caminho[b];
				copia.caminho[b] = aux; 
			}
		
			custoViabilidade(copia, fs, &viavel, &custoParcial);

			if ((copia.viavel == TRUE && viavel == TRUE && custoParcial < menorCusto) || 
				(copia.viavel == FALSE && viavel == TRUE)) {

				copia.viavel = viavel;
				menorCusto = custoParcial;
				memcpy(melhorCaminho, copia.caminho, sizeof(int) * copia.tamanhoCaminho);
				memcpy(melhorCapacidade, copia.capacidades, sizeof(int) * copia.tamanhoCaminho);
			}

			for (int a = i + 1, b = j - 1; a < b; a++, b--) { // desfazendo a reversão entre i e j
				aux = copia.caminho[a];
				copia.caminho[a] = copia.caminho[b];
				copia.caminho[b] = aux; 
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