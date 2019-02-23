#ifndef GRAFO_H
#define GRAFO_H
#include "grafo.h"
#endif

#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#ifndef VIZINHANCA_H
#define VIZINHANCA_H
#include "vizinhanca.h"
#endif

#ifndef TABU_H
#define TABU_H
#include "tabu.h"
#endif

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	char caminho[25];
	strcpy(caminho, "instancias/");
	strcat(caminho, argv[1]);
	
	Grafo g = carregarInstancia(caminho);
	FabricaSolucao fs = instanciarFabrica(g);

	Solucao s, sTralha, sAsterisco;
	int tamanhoListaTabu = 30, NbIterMax = 1000, maxSemMelhora = 80, i = 0, j = 0;

	s = instanciarSolucao(fs, construirOV_Greedy, escolherProximoVertice_Greedy);
	sAsterisco = copiarSolucao(s);
	ListaTabu tabu = criarListaTabu(tamanhoListaTabu, g.n);
	preencherListaTabu(&tabu, s.caminho, s.tamanhoCaminho);
		
	while (i <= NbIterMax) {
		sTralha = RVND(s, fs, tabu);
		liberarSolucao(s);
		s = sTralha;
		atualizarListaTabu (&tabu, s.caminho, s.tamanhoCaminho);
		if (s.custo < sAsterisco.custo) {
			liberarSolucao(sAsterisco);
			sAsterisco = copiarSolucao(s);
			j = 0;
		} else {
			j++;
			if (j == maxSemMelhora) break;
		}
		i++;
	}
	imprimirSolucao(sAsterisco, fs);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarListaTabu (tabu);
	liberarSolucao(sAsterisco);
	liberarSolucao(s);
	return 0;
}