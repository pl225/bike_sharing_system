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
	memcpy(copia.caminho, s.caminho, sizeof(int) * s.tamanhoCaminho);
	return copia;
}

Solucao supressao (Solucao s, FabricaSolucao fs) {
	Solucao nova = copiarSolucao(s);
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
		for (int i = posicaoSuprimida; i < s.tamanhoCaminho - 1; i++) nova.caminho[i] = nova.caminho[i + 1];
		nova.tamanhoCaminho--;
		nova.caminho = (int *) realloc(nova.caminho, sizeof(int) * nova.tamanhoCaminho);
		return nova;
	} else {
		return s;
	}
}

Solucao swap(Solucao s, FabricaSolucao fs) {

}