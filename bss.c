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
#include <dirent.h> 

void tabuSearch(Grafo g, FabricaSolucao fs, float results[])
{
	clock_t start = clock();

	Solucao s, sTralha, sAsterisco;
	int tamanhoListaTabu = g.n, maxSemMelhora = 100 * g.n, i = 0, j = 0, k = 0; // i = total de iterações, k = reinicios, j = iteração final

	s = instanciarSolucao(fs, construirOV_Greedy, escolherProximoVertice_Greedy);

	sAsterisco = copiarSolucao(s);
	ListaTabu tabu = criarListaTabu(tamanhoListaTabu, g.n);
	preencherListaTabu(&tabu, s.caminho, s.tamanhoCaminho);
	
	while (1) {
		sTralha = RVND(s, fs, tabu);
		liberarSolucao(s);
		s = sTralha;
		atualizarListaTabu (&tabu, s.caminho, s.tamanhoCaminho);
		if (s.custo < sAsterisco.custo) {
			liberarSolucao(sAsterisco);
			sAsterisco = copiarSolucao(s);
			j = 0;i=0;
		} else {
			j++;
			if (j == maxSemMelhora) break;
		}
		i++;
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
	srand(time(NULL));
	float results[2]; // 0 = custo, 1 = tempo
	FILE *arq = fopen(argv[1], "w");

	char caminho[50];

	DIR *d;
  	struct dirent *dir;
  	d = opendir("InstancesBRP");
  	int i = 1;

	while ((dir = readdir(d)) != NULL) {
		if (dir->d_type == DT_REG) {
			strcpy(caminho, "InstancesBRP/");
			strcat(caminho, dir->d_name);	

			Grafo g = carregarInstancia(caminho);
			FabricaSolucao fs = instanciarFabrica(g);

			fprintf(arq, "Instancia: %s, Q: %d\n", caminho, g.q);
			float mediaTempo = 0;
			float melhorCusto = INFINITY;

			for (int l = 0; l < 10; l++) {
				tabuSearch(g, fs, results);
				fprintf(arq, "\tCusto: %.f, Tempo: %f\n", results[0], results[1]);
				if (results[0] < melhorCusto) melhorCusto = results[0];
				mediaTempo += results[1];
			}
			fprintf(arq, "Melhor custo: %.f, média dos tempos: %f\n", melhorCusto, mediaTempo / 10.0);
			printf("Instancia: %s, Q: %d terminada. %d/50\n", caminho, g.q, i);

			liberarGrafo(g);
			liberarFabrica(fs);
			strcpy(caminho, "");
			i++;
		}
	}
	printf("\n\n****************TERMINADO****************\n");
	fclose(arq);
	closedir(d);
	
	return 0;
}