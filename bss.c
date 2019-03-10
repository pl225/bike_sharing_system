#include "grafo.h"
#include "solucao.h"
#include "vizinhanca.h"
#include <time.h>

void ILS_SBPRW (Grafo g, FabricaSolucao fs, FILE* arq, float results[]) {
	
	clock_t start = clock();
	
	int IR = 100;
	int iILS = 10 * g.n;
	float alpha = 0.75;
	float T0 = 1000;
	
	Solucao s, sLinha, sAsterisco, sAux;
	float T, delta, x, fAsterisco = INFINITY;
	int iterILS;
	sAux.custo = INFINITY;

	for (int i = 0; i < IR; i++) {
		s = instanciarSolucao(fs);
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
		if (sLinha.custo < fAsterisco && isViavel(sLinha)) {
			if (fAsterisco != INFINITY) liberarSolucao(sAsterisco);
			fAsterisco = sLinha.custo;
			sAsterisco = sLinha;
		} else {
			liberarSolucao(sLinha);			
		}
	}

	if (fAsterisco != INFINITY) liberarSolucao(sAsterisco);
	liberarSolucao(s);

	clock_t end = clock();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;
	
	results[0] = fAsterisco, results[1] = seconds;
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

	for (int i = 0; i < 5; i++) {
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

				fprintf(arq, "%s %d\n", caminho, qs[k]);
				double mediaTempo = 0;

				for (int l = 0; l < 10; l++) {
					ILS_SBPRW(g, fs, arq, results);
					fprintf(arq, "\tCusto%s\n", );
				}				
			}
			liberarGrafo(g);
			liberarFabrica(fs);
			strcpy(caminho, "");
		}
	}

	fclose(arq);
	return 0;
}