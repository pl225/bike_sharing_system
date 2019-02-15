#include <limits.h>
#include <stdint.h>

typedef uint32_t naco;
enum { BITS_PER_WORD = sizeof(naco) * CHAR_BIT};
#define WORD_OFFSET(b) ((b) / BITS_PER_WORD)
#define BIT_OFFSET(b) ((b) % BITS_PER_WORD)

typedef struct 
{
	naco* mapa;
	int n;
} Conjunto;

void set (Conjunto *c, int i);

void clear (Conjunto c, int i);

int get (Conjunto c, int i);

Conjunto criarConjunto (int amplitude);

void liberarConjunto (Conjunto c);
