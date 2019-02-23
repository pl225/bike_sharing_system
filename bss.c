#include "grafo.h"
#include "solucao.h"
#include "vizinhanca.h"
#include <time.h>
int main(int argc, char *argv[])
{
	char caminho[25];
	strcpy(caminho, "instancias/");
	strcat(caminho, argv[1]);
	
	Grafo g = carregarInstancia(caminho);
	int iILS = 10 * g.n;
	float alpha = 0.75;
	float T0 = 1000;
	
	srand(time(NULL));
	FabricaSolucao fs = instanciarFabrica(g);
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

	imprimirSolucao(sAsterisco, fs);
	if (sLinha.caminho != s.caminho) liberarSolucao(sLinha);
	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(sAsterisco);
	liberarSolucao(s);
	return 0;
}