#include "grafo.h"
#include "solucao.h"

int main(int argc, char const *argv[])
{
	Grafo g = carregarInstancia("n20q10A.tsp");
	FabricaSolucao fs = instanciarFabrica(g);
	Solucao s = instanciarSolucao(fs);


	liberarGrafo(g);
	liberarFabrica(fs);
	return 0;
}