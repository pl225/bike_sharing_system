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
						printf("%f\n", menorCusto);
					}
				}
			}
		}
	}
	
	if (indiceTrocaI != -1) {
		Solucao nova = copiarSolucao(s);
		int verticesMovidos[passo];
		
		for (int i = 0, j = indiceTrocaI + 1; i < passo; i++, j++) {
			verticesMovidos[i] = nova.caminho[j];
		}
		if (indiceTrocaI < indiceTrocaJ) {
			for (int i = indiceTrocaI + 1; i < indiceTrocaJ - passo; i++) {
				nova.caminho[i] = nova.caminho[i + passo];
			}
			for (int i = indiceTrocaJ - passo + 1, j = 0; j < passo; i++, j++) {
				nova.caminho[i] = verticesMovidos[j];
			}
		} else {
			for (int i = indiceTrocaI + passo; i > indiceTrocaJ + passo; i--) {
				nova.caminho[i] = nova.caminho[i - passo];
			}
			for (int i = indiceTrocaJ + 1, j = 0; j < passo; i++, j++) {
				nova.caminho[i] = verticesMovidos[j];
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