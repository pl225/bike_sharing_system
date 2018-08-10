#include "grafo.h"

int main(int argc, char const *argv[])
{
	Grafo g = carregarInstancia("n20q10A.tsp");

	liberarGrafo(g);
	return 0;
}