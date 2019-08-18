#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#ifndef CUDA_H
#define CUDA_H
#include "cuda.h"
#endif

Solucao RVND (Solucao s, FabricaSolucao fs);
Solucao _2OPT (Solucao s, FabricaSolucao fs);
Solucao swap (Solucao s, FabricaSolucao fs);
Solucao split (Solucao s, FabricaSolucao fs);
Solucao reinsercao (Solucao s, FabricaSolucao fs);
Solucao orOPT2 (Solucao s, FabricaSolucao fs);
Solucao orOPT3 (Solucao s, FabricaSolucao fs);
Solucao orOPT4 (Solucao s, FabricaSolucao fs);
