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

void tabuSearch(Grafo g, FabricaSolucao fs, float results[])
{
	clock_t start = clock();

	Solucao s, sTralha, sAsterisco;
	int tamanhoListaTabu = 20, NbIterMax = 2000, maxSemMelhora = 700, i = 0, j = 0;

	s = instanciarSolucao(fs, construirOV_Greedy, escolherProximoVertice_Greedy);

	sAsterisco = copiarSolucao(s);
	ListaTabu tabu = criarListaTabu(tamanhoListaTabu, g.n);
	preencherListaTabu(&tabu, s.caminho, s.tamanhoCaminho);
	
	while (i <= NbIterMax) {
		sTralha = RVND(s, fs, tabu);
		liberarSolucao(s);
		s = sTralha;
		atualizarListaTabu (&tabu, s.caminho, s.tamanhoCaminho);
		if (s.custo < sAsterisco.custo /*&& isViavel(s)*/) {
			liberarSolucao(sAsterisco);
			sAsterisco = copiarSolucao(s);
			j = 0;i=0;
		} else {
			j++;
			//if (j == maxSemMelhora) break;
		}
		i++;
		//s = perturbar(s, fs);
	}
	
	liberarListaTabu (tabu);
	liberarSolucao(sAsterisco);
	liberarSolucao(s);

	clock_t end = clock();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;	
	results[0] = sAsterisco.custo, results[1] = seconds;
}

int main(int argc, char *argv[])
{

	char *ns [] = {"20", "30", "40", "50", "60"};
	char *grupo [] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};
	int qs [] = {10, 15, 20, 25, 30, 35, 40, 45, 1000};
	float results[2]; // 0 = custo, 1 = tempo

	FILE *arq = fopen(argv[1], "w");

	char caminho[30];

	srand(time(NULL));

	for (int i = 1; i < 2; i++) {
		for (int j = 0; j < 10; j++) {

			strcpy(caminho, "instancias/");
			strcat(caminho, "n");
			strcat(caminho, ns[i]);
			strcat(caminho, "q10");
			strcat(caminho, grupo[j]);
			strcat(caminho, ".tsp");

			Grafo g = carregarInstancia(caminho);
			FabricaSolucao fs = instanciarFabrica(g);

			for (int k = 0; k < 9; k++) {
				
				g.q = qs[k];
				fs.q = qs[k];

				fprintf(arq, "Instancia: %s, Q: %d\n", caminho, qs[k]);
				float mediaTempo = 0;
				float melhorCusto = INFINITY;

				for (int l = 0; l < 10; l++) {
					tabuSearch(g, fs, results);
					fprintf(arq, "\tCusto: %.f, Tempo: %f\n", results[0], results[1]);
					if (results[0] < melhorCusto) melhorCusto = results[0];
					mediaTempo += results[1];
				}
				fprintf(arq, "Melhor custo: %.f, média dos tempos: %f\n", melhorCusto, mediaTempo / 10.0);
				printf("Instancia: %s, Q: %d terminada.\n", caminho, g.q);
			}
			liberarGrafo(g);
			liberarFabrica(fs);
			strcpy(caminho, "");
		}
	}
	printf("\n\n****************TERMINADO****************\n");
	fclose(arq);
	return 0;
}