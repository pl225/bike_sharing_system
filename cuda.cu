#ifndef CUDA_H
#define CUDA_H
#include "cuda.h"
#endif

#include <string.h>
#include <stdlib.h>

__device__ __forceinline__  int getPosition1D(int row, int col, int n) {
	return row * n + col;
}

__device__ __forceinline__  int linha(int r, int n) {
	return r / n;
}

__device__ __forceinline__  int coluna(int r, int n) {
	return r % n;
}

void alocarCustosGPU (FabricaSolucao fs, float** custos_gpu) {
	
	float *custos_gpu_aux;
	CHECK_ERROR(cudaMalloc(&custos_gpu_aux, fs.n * fs.n * sizeof(float)));
	CHECK_ERROR(cudaMemcpy(custos_gpu_aux, fs.custoArestas, fs.n * fs.n * sizeof(float),  cudaMemcpyHostToDevice));
	*custos_gpu = custos_gpu_aux;

}

void liberarCustosGPU (float* custos_gpu) {
	CHECK_ERROR(cudaFree(custos_gpu));
}

void alocarSolucaoGPU (Solucao s, int** caminho_gpu, ADS** ads_gpu, int** capacidade_gpu) {
	
	int *caminho_gpu_aux;
	CHECK_ERROR(cudaMalloc(&caminho_gpu_aux, s.tamanhoCaminho * sizeof(int)));
	CHECK_ERROR(cudaMemcpy(caminho_gpu_aux, s.caminho, s.tamanhoCaminho * sizeof(int),  cudaMemcpyHostToDevice));
	*caminho_gpu = caminho_gpu_aux;

    int *capacidade_gpu_aux;
    CHECK_ERROR(cudaMalloc(&capacidade_gpu_aux, s.tamanhoCaminho * sizeof(int)));
    CHECK_ERROR(cudaMemcpy(capacidade_gpu_aux, s.capacidades, s.tamanhoCaminho * sizeof(int),  cudaMemcpyHostToDevice));
    *capacidade_gpu = capacidade_gpu_aux;

	size_t tamanhoTotalADS = s.tamanhoCaminho * s.tamanhoCaminho * sizeof(ADS);
	ADS *ads_gpu_aux;
	CHECK_ERROR(cudaMalloc(&ads_gpu_aux, tamanhoTotalADS));

	ADS *ads_aux = (ADS*) malloc(tamanhoTotalADS);
	size_t tamADS = s.tamanhoCaminho * sizeof(ADS);

	for (int i = 0; i < s.tamanhoCaminho; i++) {
		memcpy(ads_aux + (i * s.tamanhoCaminho), s.ads[i], tamADS);
	}

	CHECK_ERROR(cudaMemcpy(ads_gpu_aux, ads_aux, tamanhoTotalADS,  cudaMemcpyHostToDevice));
	*ads_gpu = ads_gpu_aux;

	free(ads_aux);
}

void liberarSolucaoGPU (int* caminho_gpu, ADS* ads_gpu, int *capacidade_gpu) {
	CHECK_ERROR(cudaFree(caminho_gpu));
	CHECK_ERROR(cudaFree(ads_gpu));
    CHECK_ERROR(cudaFree(capacidade_gpu));
}

__device__ Reduzido _2OPTCuda(int posicao1D, int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu) {

	Reduzido naoViavel = {-1, -1, INFINITY};

	int i = linha(posicao1D, tamanhoCaminho), j = coluna(posicao1D, tamanhoCaminho), indiceFinal = tamanhoCaminho - 1;

	if (j - i < 3 || i >= tamanhoCaminho || j >= tamanhoCaminho) return naoViavel;
	if (ads_gpu[i].lMin > 0 || ads_gpu[i].lMax < 0) return naoViavel;

	int auxI = i + 1, auxJ = j - 1;
	ADS ads = ads_gpu[getPosition1D(auxI, auxJ, tamanhoCaminho)];
	short qSomaI = ads.qSum;
	short lMinI = -ads.qSum + ads.qMax;
	short lMaxI = ads_gpu[0].lMax - ads.qSum + ads.qMin; // ads_gpu[0].lMax é igual a fs.q

	if (ads_gpu[i].qSum >= lMinI && ads_gpu[i].qSum <= lMaxI) {
		
		short qSomaAuxiliar = ads_gpu[i].qSum + qSomaI;
		int p = getPosition1D(j, indiceFinal, tamanhoCaminho);
		if (qSomaAuxiliar >= ads_gpu[p].lMin && qSomaAuxiliar <= ads_gpu[p].lMax) {
			float custoParcial = custoOriginal - (
				custos_gpu[getPosition1D(caminho_gpu[i], caminho_gpu[auxI], tamanhoGrafo)]
					+ custos_gpu[getPosition1D(caminho_gpu[auxJ], caminho_gpu[j], tamanhoGrafo)])
				+ (custos_gpu[getPosition1D(caminho_gpu[i], caminho_gpu[auxJ], tamanhoGrafo)]
					+ custos_gpu[getPosition1D(caminho_gpu[auxI], caminho_gpu[j], tamanhoGrafo)]);

			return {i, j, custoParcial};
		}
	}

	return naoViavel;
}

__device__ Reduzido _swapCuda(int posicao1D, int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu) {

    Reduzido naoViavel = {-1, -1, INFINITY};

    int i = linha(posicao1D, tamanhoCaminho), j = coluna(posicao1D, tamanhoCaminho), indiceFinal = tamanhoCaminho - 1;

    if (i == 0 || j - i < 1 || i >= tamanhoCaminho - 1 || j >= tamanhoCaminho - 1) return naoViavel;
    if (ads_gpu[i - 1].lMin > 0 || ads_gpu[i - 1].lMax < 0) return naoViavel;

    int p = getPosition1D(j, j, tamanhoCaminho);

    if (ads_gpu[i - 1].qSum >= ads_gpu[p].lMin && ads_gpu[i - 1].qSum <= ads_gpu[p].lMax) {
        
        short qSumAuxiliar = ads_gpu[i - 1].qSum + ads_gpu[p].qSum;
        if ((j - 1) - (i + 1) >= 0) {
            int a = i + 1, b = j - 1;
            p = getPosition1D(a, b, tamanhoCaminho);
            if (qSumAuxiliar >= ads_gpu[p].lMin && qSumAuxiliar <= ads_gpu[p].lMax)
                qSumAuxiliar += ads_gpu[p].qSum;
            else 
                return naoViavel;
        }

        p = getPosition1D(i, i, tamanhoCaminho);
        if (qSumAuxiliar >= ads_gpu[p].lMin && qSumAuxiliar <= ads_gpu[p].lMax) {
            
            qSumAuxiliar += ads_gpu[p].qSum;
            p = getPosition1D(j + 1, indiceFinal, tamanhoCaminho);
            
            if (qSumAuxiliar >= ads_gpu[p].lMin && qSumAuxiliar <= ads_gpu[p].lMax) {

                float custoAux = 0, custoAuxAntigo = 0, menorCustoParcial = 0;

                if (j - i > 1) {
                    custoAux = custos_gpu[getPosition1D(caminho_gpu[i - 1], caminho_gpu[j], tamanhoGrafo)] 
                        + custos_gpu[getPosition1D(caminho_gpu[j], caminho_gpu[i + 1], tamanhoGrafo)];
                    custoAux += custos_gpu[getPosition1D(caminho_gpu[j - 1], caminho_gpu[i], tamanhoGrafo)] 
                        + custos_gpu[getPosition1D(caminho_gpu[i], caminho_gpu[j + 1], tamanhoGrafo)];

                    custoAuxAntigo = custos_gpu[getPosition1D(caminho_gpu[i - 1], caminho_gpu[i], tamanhoGrafo)] 
                        + custos_gpu[getPosition1D(caminho_gpu[i], caminho_gpu[i + 1], tamanhoGrafo)];
                    custoAuxAntigo += custos_gpu[getPosition1D(caminho_gpu[j - 1], caminho_gpu[j], tamanhoGrafo)] 
                        + custos_gpu[getPosition1D(caminho_gpu[j], caminho_gpu[j + 1], tamanhoGrafo)];
                } else {
                    custoAux = custos_gpu[getPosition1D(caminho_gpu[i - 1], caminho_gpu[j], tamanhoGrafo)] 
                        + custos_gpu[getPosition1D(caminho_gpu[i], caminho_gpu[j + 1], tamanhoGrafo)];
                    custoAuxAntigo = custos_gpu[getPosition1D(caminho_gpu[i - 1], caminho_gpu[i], tamanhoGrafo)] 
                        + custos_gpu[getPosition1D(caminho_gpu[j], caminho_gpu[j + 1], tamanhoGrafo)];
                }

                menorCustoParcial = custoOriginal + custoAux - custoAuxAntigo;
                return {i, j, menorCustoParcial};
            }
        }
    }

    return naoViavel;
}

__device__ Reduzido _orOPTCuda(int posicao1D, int tamanhoCaminho, int tamanhoGrafo, 
                              float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, int tipo) {

    Reduzido naoViavel = {-1, -1, INFINITY};
    int passo, condicaoParada;

    if (tipo == 0) { // reinsercao
        passo = 0;
        condicaoParada = tamanhoCaminho - 2;
    } else if (tipo == 1) { // orOPT2
        passo = 1;
        condicaoParada = tamanhoCaminho - 3;
    } else if (tipo == 2) { // orOPT3
        passo = 2;
        condicaoParada = tamanhoCaminho - 4;
    } else { // orOPT4
        passo = 3;
        condicaoParada = tamanhoCaminho - 5;
    }

    int i = linha(posicao1D, tamanhoCaminho), j = coluna(posicao1D, tamanhoCaminho);

    if (i == 0 || i == j || i >= condicaoParada || j >= tamanhoCaminho - 1) return naoViavel;
    if (i < j && j - i < passo + 1) return naoViavel; // deve haver uma subsequência de tamanho >= passo + 1 // i == j : j += passo + 1
    if (i > j && i - j < 2) return naoViavel; // para os casos em q i está na frente de j

    int fimSeg4 = tamanhoCaminho - 1, fimSeg1, iniSeg2, fimSeg2, iniSeg3, fimSeg3, iniSeg4;

    if (i < j) {
        fimSeg1 = i - 1, iniSeg2 = i + passo + 1, fimSeg2 = j, iniSeg3 = i,
            fimSeg3 = i + passo, iniSeg4 = j + 1;
    } else {
        fimSeg1 = j, iniSeg2 = i, fimSeg2 = i + passo, iniSeg3 = j + 1,
            fimSeg3 = i - 1, iniSeg4 = i + passo + 1;
    }

    if (ads_gpu[fimSeg1].lMin > 0 || ads_gpu[fimSeg1].lMax < 0) return naoViavel;

    short qSumAuxiliar;
    float menorCustoParcial = custoOriginal - (custos_gpu[getPosition1D(caminho_gpu[i - 1], caminho_gpu[i], tamanhoGrafo)]
                + custos_gpu[getPosition1D(caminho_gpu[i + passo], caminho_gpu[i + passo + 1], tamanhoGrafo)]);

    menorCustoParcial += custos_gpu[getPosition1D(caminho_gpu[i - 1], caminho_gpu[i + passo + 1], tamanhoGrafo)];

    int p = getPosition1D(iniSeg2, fimSeg2, tamanhoCaminho);
    if (ads_gpu[fimSeg1].qSum >= ads_gpu[p].lMin && ads_gpu[fimSeg1].qSum <= ads_gpu[p].lMax) {
    
        qSumAuxiliar = ads_gpu[fimSeg1].qSum + ads_gpu[p].qSum;
        p = getPosition1D(iniSeg3, fimSeg3, tamanhoCaminho);
        if (qSumAuxiliar >= ads_gpu[p].lMin && qSumAuxiliar <= ads_gpu[p].lMax) {
        
            qSumAuxiliar += ads_gpu[p].qSum;
            p = getPosition1D(iniSeg4, fimSeg4, tamanhoCaminho);
            if (qSumAuxiliar >= ads_gpu[p].lMin && qSumAuxiliar <= ads_gpu[p].lMax) {
                
                menorCustoParcial += custos_gpu[getPosition1D(caminho_gpu[j], caminho_gpu[i], tamanhoGrafo)]
                + custos_gpu[getPosition1D(caminho_gpu[i + passo], caminho_gpu[j + 1], tamanhoGrafo)];
                
                menorCustoParcial -= custos_gpu[getPosition1D(caminho_gpu[j], caminho_gpu[j + 1], tamanhoGrafo)];
                
                return {i, j, menorCustoParcial};
            }
        }
    }
    
    return naoViavel;
}

__device__ Reduzido _splitCuda(int posicao1D, int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, int *capacidade_gpu) {

    Reduzido naoViavel = {-1, -1, INFINITY};

    int i = linha(posicao1D, tamanhoCaminho), j = coluna(posicao1D, tamanhoCaminho), indiceFinal = tamanhoCaminho - 1;

    if (i == 0 || i == j || i >= tamanhoCaminho - 1 || j >= tamanhoCaminho - 1) return naoViavel;
    if (abs(capacidade_gpu[i] - capacidade_gpu[i - 1]) <= 1) return naoViavel;

    short qSum, lMin2, lMax2, qSum2, lMin4, lMax4, qSum4;
    int fimSeg1, iniSeg3, fimSeg3, iniSeg5, p;

    if (capacidade_gpu[i] - capacidade_gpu[i - 1] < - 1) { // coleta
        p = getPosition1D(i, i, tamanhoCaminho);
        qSum = ads_gpu[p].qSum - 1;
        if (i < j) {
            lMin4 = 0, lMax4 = ads_gpu[0].lMax - 1, qSum4 = 1,
                qSum2 = qSum, lMin2 = 0, lMax2 = ads_gpu[0].lMax - qSum;
        } else {
            lMin2 = 0, lMax2 = ads_gpu[0].lMax - 1, qSum2 = 1,
                qSum4 = qSum, lMin4 = 0, lMax4 = ads_gpu[0].lMax - qSum;
        }
    } else { // entrega
        p = getPosition1D(i, i, tamanhoCaminho);
        qSum = ads_gpu[p].qSum + 1;
        if (i < j) {
            lMin4 = 1, lMax4 = ads_gpu[0].lMax, qSum4 = -1,
                qSum2 = qSum, lMin2 = -qSum, lMax2 = ads_gpu[0].lMax;
        } else {
            lMin2 = 1, lMax2 = ads_gpu[0].lMax, qSum2 = -1,
                qSum4 = qSum, lMin4 = -qSum, lMax4 = ads_gpu[0].lMax;
        }
    }

    if (i < j) {
        fimSeg1 = i - 1, iniSeg3 = i + 1, fimSeg3 = j - 1, iniSeg5 = j;
    } else {
        fimSeg1 = j - 1, iniSeg3 = j, fimSeg3 = i - 1, iniSeg5 = i + 1;
    }

    if (ads_gpu[fimSeg1].lMin > 0 || ads_gpu[fimSeg1].lMax < 0) return naoViavel;

    if (ads_gpu[fimSeg1].qSum >= lMin2 && ads_gpu[fimSeg1].qSum <= lMax2) {
        
        qSum = ads_gpu[fimSeg1].qSum + qSum2;
        p = getPosition1D(iniSeg3, fimSeg3, tamanhoCaminho);

        if (qSum >= ads_gpu[p].lMin && qSum <= ads_gpu[p].lMax) {
            qSum += ads_gpu[p].qSum;
            
            if (qSum >= lMin4 && qSum <= lMax4) {
                
                qSum += qSum4;
                p = getPosition1D(iniSeg5, indiceFinal, tamanhoCaminho);

                if (qSum >= ads_gpu[p].lMin && qSum <= ads_gpu[p].lMax) {

                    float custoParcial = custoOriginal - custos_gpu[getPosition1D(caminho_gpu[j - 1], caminho_gpu[j], tamanhoGrafo)]
                        + (custos_gpu[getPosition1D(caminho_gpu[j - 1], caminho_gpu[i], tamanhoGrafo)]
                        + custos_gpu[getPosition1D(caminho_gpu[i], caminho_gpu[j], tamanhoGrafo)]);

                    return {i, j, custoParcial};
                }
            }
        }
    }
    
    return naoViavel;
}

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int* blocks, int* threads) {

    cudaDeviceProp prop;
    int device;
    CHECK_ERROR(cudaGetDevice(&device));
    CHECK_ERROR(cudaGetDeviceProperties(&prop, device));

    *threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
    *blocks = (n + *threads - 1) / *threads;

    if (*blocks > prop.maxGridSize[0]) {
        //printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
          //     blocks, prop.maxGridSize[0], *threads * 2, *threads);

        *blocks /= 2;
        *threads *= 2;
    }

}

__device__ Reduzido escolherVizinhanca(int posicao1D, 
                                                int tamanhoCaminho, 
                                                int tamanhoGrafo, 
                                                float custoOriginal, 
                                                int *caminho_gpu, 
                                                ADS *ads_gpu, 
                                                float *custos_gpu,  
                                                int* capacidade_gpu, 
                                                Vizinhanca v) {
    switch (v) {
        case _2OPT_GPU:
            return _2OPTCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu);
        case SWAP_GPU:
            return _swapCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu);
        case SPLIT_GPU:
            return _splitCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu);
        case OROPT1_GPU:
            return _orOPTCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu, 0);
        case OROPT2_GPU:
            return _orOPTCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu, 1);
        case OROPT3_GPU:
            return _orOPTCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu, 2);
        case OROPT4_GPU:
            return _orOPTCuda(posicao1D, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu, 3);
        default:
            return {-1, -1, INFINITY};
    }
}

__global__ void reduzir(int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, int *capacidade_gpu, Reduzido* g_idata, Reduzido* g_odata, 
						int size, int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, Vizinhanca v) {
    
    extern __shared__ Reduzido sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (g_idata == NULL)
        sdata[tid] = escolherVizinhanca(i, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu, v);
    else if (i < size) 
        sdata[tid] = g_idata[i];
    else
        sdata[tid] = {-1, -1, INFINITY};
    
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s>0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid].custo > sdata[tid + s].custo)
                sdata[tid] = sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void preparadorReducao(int size, int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int threads, int blocks, 
						int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, int *capacidade_gpu, Reduzido *d_idata, Reduzido *d_odata, Vizinhanca v) {
    
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(Reduzido) : threads * sizeof(Reduzido);

	reduzir<<< dimGrid, dimBlock, smemSize >>>(caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu, d_idata, d_odata, size, tamanhoCaminho, tamanhoGrafo, custoOriginal, v);
}

Reduzido reducaoAuxiliar(int  n,
				  int tamanhoCaminho,
				  int tamanhoGrafo,
				  float custoOriginal,
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  int *caminho_gpu, 
                  ADS *ads_gpu, 
                  float *custos_gpu,
                  int *capacidade_gpu, 
                  Reduzido *d_idata, 
                  Reduzido *d_odata,
                  Reduzido *h_odata,
                  Vizinhanca v) {

    Reduzido gpu_result = {-1, -1, INFINITY};
    bool needReadBack = true;
    int  cpuFinalThreshold = 1;

    cudaDeviceSynchronize();

    preparadorReducao(n, tamanhoCaminho, tamanhoGrafo, custoOriginal, numThreads, numBlocks, caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu, NULL, d_odata, v);
    
    int s = numBlocks;

    while (s > cpuFinalThreshold) {
        
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, &blocks, &threads);
        cudaMemcpy(d_idata, d_odata, s * sizeof(Reduzido), cudaMemcpyDeviceToDevice);
        preparadorReducao(s, tamanhoCaminho, tamanhoGrafo, custoOriginal, threads, blocks, caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu, d_idata, d_odata, v);

        s = (s + threads - 1) / threads;
    }

    if (s > 1) {
        CHECK_ERROR(cudaMemcpy(h_odata, d_odata, s * sizeof(Reduzido), cudaMemcpyDeviceToHost));
        for (int i = 0; i < s; i++) {
            if (gpu_result.custo > h_odata[i].custo)
                gpu_result = h_odata[i];
        }

        needReadBack = false;
    }

    cudaDeviceSynchronize();

    if (needReadBack) {
        CHECK_ERROR(cudaMemcpy(&gpu_result, d_odata, sizeof(Reduzido), cudaMemcpyDeviceToHost));
    }

    return gpu_result;
}

Reduzido runTest(int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, int* capacidade_gpu, Vizinhanca v) {
    
    int size = tamanhoCaminho * tamanhoCaminho;    // number of elements to reduce
    int maxThreads = 256;  // number of threads per block
    int maxBlocks = 64;
    int numBlocks = 0;
    int numThreads = 0;

    //printf("%d elements\n", size);
    //printf("%d threads (max)\n", maxThreads);

    getNumBlocksAndThreads(size, maxBlocks, maxThreads, &numBlocks, &numThreads);

    // allocate mem for the result on host side
    Reduzido *h_odata = (Reduzido *) malloc(numBlocks * sizeof(Reduzido));

    int bytes = size * sizeof(Reduzido);
    Reduzido *d_odata = NULL, *d_idata = NULL;
	CHECK_ERROR(cudaMalloc((void **) &d_odata, numBlocks * sizeof(Reduzido))); // saída tem o tamanho do número de blocos
	CHECK_ERROR(cudaMalloc((void **) &d_idata, bytes));

    //printf("%d blocks\n\n", numBlocks);


    Reduzido gpu_result = reducaoAuxiliar(size, tamanhoCaminho, tamanhoGrafo, custoOriginal, numThreads, numBlocks, maxThreads, maxBlocks,
                                    caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu, d_idata, d_odata, h_odata, v);

    free(h_odata);
    CHECK_ERROR(cudaFree(d_odata));
    CHECK_ERROR(cudaFree(d_idata));

    return gpu_result;
}

Reduzido obterVizinho (Solucao s, FabricaSolucao fs, float* custos_gpu, Vizinhanca v) {

    //float *custos_gpu = NULL;
    int *caminho_gpu = NULL;
    int *capacidade_gpu = NULL;
    ADS *ads_gpu = NULL;

    //alocarCustosGPU(fs, &custos_gpu);
    alocarSolucaoGPU (s, &caminho_gpu, &ads_gpu, &capacidade_gpu);
    
    Reduzido r = runTest(s.tamanhoCaminho, fs.n, s.custo, caminho_gpu, ads_gpu, custos_gpu, capacidade_gpu, v);
    CHECK_ERROR(cudaDeviceSynchronize());

    //liberarCustosGPU(custos_gpu);
    liberarSolucaoGPU(caminho_gpu, ads_gpu, capacidade_gpu);

    return r;

}