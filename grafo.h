#define N_COLUNAS 2

typedef struct Grafo
{
	int n;
	int q;
	float *custos;
	int *demandas;
} Grafo;

void liberarGrafo (Grafo g);

Grafo carregarInstancia (char caminhoArquivo []);