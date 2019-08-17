#include "grafo.h"
#include "solucao.h"
#include "vizinhanca.h"
#include <time.h>
#include "cuda.h"

#define GPU 0

__global__ void gpu(int *caminho_gpu, ADS *ads_gpu, float *custos_gpu) {
	printf("%hi %hi %hi %hi %hi\n", ads_gpu[0].qSum, ads_gpu[0].qMin, ads_gpu[0].qMax, ads_gpu[0].lMin, ads_gpu[0].lMax);   
	printf("%f\n", custos_gpu[0]);
	printf("%d\n", caminho_gpu[0]);
}

int main(int argc, char *argv[])
{

	CHECK_ERROR(cudaSetDevice(GPU));
	CHECK_ERROR(cudaDeviceReset());

	//srand(time(NULL));

	Grafo g = carregarInstancia("instancias/n500q1000A.tsp");
	FabricaSolucao fs = instanciarFabrica(g);
	Solucao s = instanciarSolucao(fs);

	float *custos_gpu = NULL;
	int *caminho_gpu = NULL;
	ADS *ads_gpu = NULL;

	alocarCustosGPU(fs, &custos_gpu);
	alocarSolucaoGPU (s, &caminho_gpu, &ads_gpu);
	printf("%d %.f\n", s.tamanhoCaminho, s.custo);
	//gpu<<<1, 1>>>(caminho_gpu, ads_gpu, custos_gpu);
	runTest(s.tamanhoCaminho, g.n, s.custo, caminho_gpu, ads_gpu, custos_gpu);
   	CHECK_ERROR(cudaDeviceSynchronize());

	liberarCustosGPU(custos_gpu);
	liberarSolucaoGPU(caminho_gpu, ads_gpu);

	
	/*clock_t start, end;
	start = clock();
	Solucao s = instanciarSolucao(fs);
	end = clock();

	double t = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Construcao: %lf\n", t);

	//imprimirSolucao(s, fs);

	start = clock();
	Solucao ss = RVND(s, fs);
	end = clock();

	t = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("RVND: %lf\n", t);

	//imprimirSolucao(ss, fs);

	liberarGrafo(g);
	liberarFabrica(fs);
	liberarSolucao(s);
	liberarSolucao(ss);*/

	return 0;
}