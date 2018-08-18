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