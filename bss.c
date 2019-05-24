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

	Solucao s, sTralha, sAsterisco, sLinha;
	sAsterisco.custo = INFINITY;
	int tamanhoListaTabu = g.n, maxSemMelhora = 10 * g.n, i = 0, j = 0, k = 0, IR = 100; // i = total de iterações, k = reinicios, j = iteração final

	for (int a = 0; a < IR; a++) {
		s = instanciarSolucao(fs, construirOV_ILS_RVND, escolherProximoVertice_ILS_RVND);

		sLinha = copiarSolucao(s);
		ListaTabu tabu = criarListaTabu(tamanhoListaTabu, g.n);
		preencherListaTabu(&tabu, s.caminho, s.tamanhoCaminho);
		
		while (1) {
			sTralha = RVND(s, fs, tabu);
			liberarSolucao(s);
			s = sTralha;
			atualizarListaTabu (&tabu, s.caminho, s.tamanhoCaminho);
			if (s.custo < sLinha.custo) {
				liberarSolucao(sLinha);
				sLinha = copiarSolucao(s);
				j = 0; k++;
			} else {
				j++;
				if (j == maxSemMelhora) break;
			}
			i++;
		}

		if (sLinha.custo < sAsterisco.custo) {
			if (sAsterisco.custo != INFINITY) liberarSolucao(sAsterisco);
			sAsterisco = sLinha;
		} else {
			liberarSolucao(sLinha);			
		}
		liberarSolucao(s);
		liberarListaTabu(tabu);
	}
	
	liberarSolucao(sAsterisco);

	clock_t end = clock();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;	
	results[0] = sAsterisco.custo, results[1] = seconds, results[2] = i, results[3] = j, results[4] = k;
}

int main(int argc, char *argv[])
{

	char *ns [] = {"20", "30", "40", "50", "60"};
	char *grupo [] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};
	int qs [] = {10, 15, 20, 25, 30, 35, 40, 45, 1000};
	float results[5]; // 0 = custo, 1 = tempo

	FILE *arq = fopen(argv[1], "w");

	char caminho[30];

	srand(time(NULL));

	for (int i = 4; i < 5; i++) {
		for (int j = 6; j < 10; j++) {

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
					fprintf(arq, "\tCusto: %.f, Tempo: %f, Total de iteracoes: %.f, Iteracao final: %.f, Reinicios: %.f\n", 
						results[0], results[1], results[2], results[3], results[4]);
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