#include "grafo.h"
#include "solucao.h"
#include <time.h>

int main(int argc, char const *argv[])
{
	srand(time(NULL));
	Grafo g = carregarInstancia("n20q10A.tsp");
	FabricaSolucao fs = instanciarFabrica(g);
	Solucao s = instanciarSolucao(fs);


	liberarGrafo(g);
	liberarFabrica(fs);
	return 0;
}