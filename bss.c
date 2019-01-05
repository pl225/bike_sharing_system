#include "grafo.h"
#include "solucao.h"
#include "vizinhanca.h"
#include <time.h>

int main(int argc, char *argv[])
{
	int IR = atoi(argv[2]);
	int iILS = atoi(argv[3]);
	int alpha = atof(argv[4]);
	float T0 = atof(argv[5]);
	
	srand(time(NULL));
	Grafo g = carregarInstancia(argv[1]); //"n20q10A.tsp"
	FabricaSolucao fs = instanciarFabrica(g);
	Solucao s, sLinha, sAsterisco;
	float T, delta, x, fAsterisco = INFINITY;
	int iterILS;

	for (int i = 0; i < IR; i++) {
		s = instanciarSolucao(fs);
		sLinha = s; //copia
		T = T0;
		iterILS = 0;
		
		while (iterILS < iILS) {
			s = RVND(s, fs);// antes de executar o RVND, s == sLinha
			delta = s.custo - sLinha.custo;
			if (delta < 0) {
				sLinha = s;//antes, liberar sLinha
				iterILS = 0;
			} else {
				x = ((double) rand() / (RAND_MAX));
				if (T > 0 && x < exp(-(delta / T))) {
					sLinha = s; //antes, liberar sLinha se delta nao for zero
				}
			}
			s = perturbar(sLinha, fs);
			iterILS += 1;
			T *= alpha;
		}
		if (sLinha.custo < fAsterisco) {
			fAsterisco = sLinha.custo;
			sAsterisco = sLinha; //apos, liberar sLinha 
		}
	}

	imprimirSolucao(sAsterisco);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(s);
	liberarSolucao(sAsterisco);
	return 0;
}