#include "grafo.h"
#include "solucao.h"
#include "vizinhanca.h"
#include <time.h>
int main(int argc, char *argv[])
{
	Grafo g = carregarInstancia("n20q10A.tsp");
	int IR = 100;
	int iILS = 10 * g.n;
	float alpha = 0.75;
	float T0 = 1000;
	
	srand(time(NULL));
	FabricaSolucao fs = instanciarFabrica(g);
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
		if (sLinha.custo < fAsterisco) {
			if (fAsterisco != INFINITY) liberarSolucao(sAsterisco);
			fAsterisco = sLinha.custo;
			sAsterisco = sLinha;
		} else {
			liberarSolucao(sLinha);			
		}
	}

	imprimirSolucao(sAsterisco, fs);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(sAsterisco);
	liberarSolucao(s);
	return 0;
}