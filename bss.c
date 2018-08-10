#include "grafo.h"

int main(int argc, char const *argv[])
{
	Grafo g = carregarInstancia("n20q10A.tsp");

	printf("%f\n", g.pontos[4*2 + 1]);
	return 0;
}