#include "grafo.h"
#include "solucao.h"
#include "vizinhanca.h"
#include <time.h>
#include <dirent.h> 

void ILS_SBPRW (Grafo g, FabricaSolucao fs, FILE* arq, float results[]) {
	
	clock_t start = clock();
	
	int iILS = 10 * g.n;
	float alpha = 0.75;
	float T0 = 1000;
	
	Solucao s, sLinha, sAsterisco, sAux;
	float T, delta, x, fAsterisco = INFINITY;
	int iterILS;
	sAux.custo = INFINITY;

	s = instanciarSolucao(fs, construirOV_Greedy, escolherProximoVertice_Greedy);
	sLinha = s;
	T = T0;
	iterILS = 0;
		
	while (iterILS < iILS) {
		s = RVND(s, fs);
		delta = s.custo - sLinha.custo;
		if (sAux.custo != INFINITY && s.caminho != sAux.caminho){ liberarSolucao(sAux);}
		if (delta < 0) {
			liberarSolucao(sLinha);
			sLinha = s;
			iterILS = 0;
			if (s.custo < fAsterisco && isViavel(s)) {
				if (fAsterisco != INFINITY) liberarSolucao(sAsterisco);
				fAsterisco = s.custo;
				sAsterisco = copiarSolucao(s);
			}
		} else {
			x = ((double) rand() / (RAND_MAX));
			if (T > 0 && x < exp(-(delta / T))) {
				if (s.caminho != sLinha.caminho) liberarSolucao(sLinha);
				sLinha = s;
			} else if (s.caminho != sLinha.caminho) {
				liberarSolucao(s);
			}
		}
		s = perturbar(sLinha, fs);
		sAux = s;
		iterILS += 1;
		T *= alpha;
	}

	if (fAsterisco != INFINITY) liberarSolucao(sAsterisco);
	liberarSolucao(s);

	clock_t end = clock();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;
	
	results[0] = fAsterisco, results[1] = seconds;
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
				ILS_SBPRW(g, fs, arq, results);
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

			if (i == 3) break;
		}
	}
	printf("\n\n****************TERMINADO****************\n");
	fclose(arq);
	closedir(d);
	
	return 0;
}