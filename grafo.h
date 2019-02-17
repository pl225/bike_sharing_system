#define N_COLUNAS 2

typedef struct Grafo
{
	int n;
	int q;
	int *pontos;
	int *demandas;
} Grafo;

void liberarGrafo (Grafo g);

int IndicePontos (int i, int j);

Grafo carregarInstancia (char caminhoArquivo []);