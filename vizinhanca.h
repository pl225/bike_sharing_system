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
	copia.caminho = (int *) malloc(sizeof(int) * s.tamanhoCaminho);
	copia.capacidades = (int *) malloc(sizeof(int) * s.tamanhoCaminho);
	copia.viavel = s.viavel;
	memcpy(copia.caminho, s.caminho, sizeof(int) * s.tamanhoCaminho);
	memcpy(copia.capacidades, s.capacidades, sizeof(int) * s.tamanhoCaminho);
	return copia;
}

Solucao supressao (Solucao s, FabricaSolucao fs) {
	float menorCusto = custo(s, fs), ultimoCustoRemovido = 0, menorCustoParcial, custoAux;
	int posicaoSuprimida = -1;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		if (fs.demandas[s.caminho[i]] == 0) {
			custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
				+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
			menorCustoParcial = menorCusto - custoAux + fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i + 1], fs.n)];
			if (menorCustoParcial + ultimoCustoRemovido <= menorCusto) {
				posicaoSuprimida = i;
				menorCusto = menorCustoParcial + ultimoCustoRemovido;
				ultimoCustoRemovido = custoAux;
			}
		}
	}

	if (posicaoSuprimida != -1) {
		Solucao nova = copiarSolucao(s);
		for (int i = posicaoSuprimida; i < s.tamanhoCaminho - 1; i++) {
			nova.caminho[i] = nova.caminho[i + 1];
			nova.capacidades[i] = nova.capacidades[i + 1];
		}
		nova.tamanhoCaminho--;
		nova.caminho = (int *) realloc(nova.caminho, sizeof(int) * nova.tamanhoCaminho);
		nova.capacidades = (int *) realloc(nova.capacidades, sizeof(int) * nova.tamanhoCaminho);
		return nova;
	} else {
		return s;
	}
}

Solucao swap(Solucao s, FabricaSolucao fs) {
	float menorCusto = custo(s, fs), custoOriginal = menorCusto, menorCustoParcial, custoAux = 0,	custoAuxAntigo = 0;
	int indiceTrocaI = -1, indiceTrocaJ = -1;

	for (int i = 1; i < s.tamanhoCaminho - 1; i++) {
		for (int j = i + 1; j < s.tamanhoCaminho - 1; j++) {
			if (s.capacidades[i - 1] == s.capacidades[j - 1] && s.capacidades[i] == s.capacidades[j]) {
				
				custoAux = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[j], fs.n)] 
					+ fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[i + 1], fs.n)];
				custoAux += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[i], fs.n)] 
					+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[j + 1], fs.n)];

				custoAuxAntigo = fs.custoArestas[IndiceArestas(s.caminho[i - 1], s.caminho[i], fs.n)] 
					+ fs.custoArestas[IndiceArestas(s.caminho[i], s.caminho[i + 1], fs.n)];
				custoAuxAntigo += fs.custoArestas[IndiceArestas(s.caminho[j - 1], s.caminho[j], fs.n)] 
					+ fs.custoArestas[IndiceArestas(s.caminho[j], s.caminho[j + 1], fs.n)];

				menorCustoParcial = custoOriginal + custoAux - custoAuxAntigo;
				if (menorCustoParcial <= menorCusto) {
					indiceTrocaI = i;
					indiceTrocaJ = j;
					menorCusto = menorCustoParcial;
				} 
			}
		}
	}

	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		int aux = nova.caminho[indiceTrocaI];
		nova.caminho[indiceTrocaI] = nova.caminho[indiceTrocaJ];
		nova.caminho[indiceTrocaJ] = aux;
		return nova;
	} else {
		return s;
	}
}

/*Solucao reinsercao(Solucao s, FabricaSolucao fs) {
	printf("%d\n", s.tamanhoCaminho);
	for (int i = 1; i < s.tamanhoCaminho - 2; i ++) {
		if ((s.capacidades[i - 1] - abs(s.capacidades[i + 1] - s.capacidades[i])) < 0 
			|| (s.capacidades[i - 1] + abs(s.capacidades[i + 1] - s.capacidades[i])) > fs.q)
			continue;
		printf("%d %d\n", i, s.capacidades[i]);
	}
}*/

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