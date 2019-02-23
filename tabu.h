#include "bitmap.h"

typedef struct
{
	int i, j, k; // a(i, j) e posicao k
} ArcoPosicao;

typedef struct 
{
	ArcoPosicao* feitos;
	int n;
} Movimento;

typedef struct
{
	Conjunto** arcos;
	Movimento* movimentos;
	int tamanho;
	int nVertices;
	int indiceAtual;
} ListaTabu;

ListaTabu criarListaTabu (int n, int nVertices);

void liberarListaTabu (ListaTabu lista);

void preencherListaTabu (ListaTabu* lista, int* caminho, int tamanhoCaminho);

void atualizarListaTabu (ListaTabu* lista, int* caminho, int tamanhoCaminho);

int tabuContem(ListaTabu lista, int i, int j, int k);