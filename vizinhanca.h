#ifndef SOLUCAO_H
#define SOLUCAO_H
#include "solucao.h"
#endif

#ifndef TABU_H
#define TABU_H
#include "tabu.h"
#endif

Solucao perturbar (Solucao s, FabricaSolucao fs);

//Solucao RVND (Solucao s, FabricaSolucao fs, ListaTabu lista);

Solucao exchange_2_2(Solucao s, FabricaSolucao fs);
Solucao exchange_1_2(Solucao s, FabricaSolucao fs);
Solucao split(Solucao s, FabricaSolucao fs);
Solucao splitP(Solucao s, FabricaSolucao fs);
Solucao _3OPT_P(Solucao s, FabricaSolucao fs);
Solucao _3OPT(Solucao s, FabricaSolucao fs);