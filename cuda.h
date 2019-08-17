#define CHECK_ERROR(call) do {                                                    \
	if(cudaSuccess != call) {                                                     \
		fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", cudaGetErrorString(call),  __FILE__, __LINE__); \
		exit(0);                                                                                 \
    } } while (0)

typedef struct Reduzido
{
	int i, j;
	float custo;
} Reduzido;

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

void alocarSolucaoGPU (Solucao s, int** caminho_gpu, ADS** ads_gpu) {
	
	int *caminho_gpu_aux;
	CHECK_ERROR(cudaMalloc(&caminho_gpu_aux, s.tamanhoCaminho * sizeof(int)));
	CHECK_ERROR(cudaMemcpy(caminho_gpu_aux, s.caminho, s.tamanhoCaminho * sizeof(int),  cudaMemcpyHostToDevice));
	*caminho_gpu = caminho_gpu_aux;

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

void liberarSolucaoGPU (int* caminho_gpu, ADS* ads_gpu) {
	CHECK_ERROR(cudaFree(caminho_gpu));
	CHECK_ERROR(cudaFree(ads_gpu));
}

__device__ Reduzido _2OPTCuda (int posicao1D, int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu) {

	Reduzido naoViavel;
	naoViavel.i = 0, naoViavel.i = 0, naoViavel.custo = INFINITY;

	int i = linha(posicao1D, tamanhoCaminho), j = coluna(posicao1D, tamanhoCaminho), indiceFinal = tamanhoCaminho - 1;

	if (j - i < 3 || i >= tamanhoCaminho) return naoViavel;
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

			Reduzido r;
			r.i = i, r.j = j, r.custo = custoParcial;
			return r;
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
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], *threads * 2, *threads);

        *blocks /= 2;
        *threads *= 2;
    }

}

__global__ void reduzir(int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, Reduzido* g_idata, Reduzido* g_odata, 
						int tamanhoCaminho, int tamanhoGrafo, float custoOriginal) {
    
    extern __shared__ Reduzido sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = _2OPTCuda(i, tamanhoCaminho, tamanhoGrafo, custoOriginal, caminho_gpu, ads_gpu, custos_gpu);

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
						int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, Reduzido *d_idata, Reduzido *d_odata) {
    
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(Reduzido) : threads * sizeof(Reduzido);

	reduzir<<< dimGrid, dimBlock, smemSize >>>(caminho_gpu, ads_gpu, custos_gpu, d_idata, d_odata, tamanhoCaminho, tamanhoGrafo, custoOriginal);
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
                  Reduzido *d_idata, 
                  Reduzido *d_odata,
                  Reduzido *h_odata) {

    Reduzido gpu_result;
    gpu_result.i = - 1, gpu_result.j = -1, gpu_result.custo = INFINITY;
    bool needReadBack = true;
    int  cpuFinalThreshold = 1;

    cudaDeviceSynchronize();

    preparadorReducao(n, tamanhoCaminho, tamanhoGrafo, custoOriginal, numThreads, numBlocks, caminho_gpu, ads_gpu, custos_gpu, d_idata, d_odata);
    
    int s = numBlocks;

    /*while (s > cpuFinalThreshold) {
        
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, &blocks, &threads);
        cudaMemcpy(d_idata, d_odata, s * sizeof(Reduzido), cudaMemcpyDeviceToDevice);
        reduzir(s, threads, blocks, kernel, caminho_gpu, ads_gpu, custos_gpu, d_odata);

        s = (s + threads - 1) / threads;
    }*/

    if (/*s > 1*/1) {
        // copy result from device to host
        CHECK_ERROR(cudaMemcpy(h_odata, d_odata, s * sizeof(Reduzido), cudaMemcpyDeviceToHost));
        for (int i=0; i < s; i++) {
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

void runTest(int tamanhoCaminho, int tamanhoGrafo, float custoOriginal, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu) {
    
    int size = tamanhoCaminho * tamanhoCaminho;    // number of elements to reduce
    int maxThreads = 256;  // number of threads per block
    int maxBlocks = 64;
    int numBlocks = 0;
    int numThreads = 0;

    printf("%d elements\n", size);
    printf("%d threads (max)\n", maxThreads);

    getNumBlocksAndThreads(size, maxBlocks, maxThreads, &numBlocks, &numThreads);

    // allocate mem for the result on host side
    Reduzido *h_odata = (Reduzido *) malloc(numBlocks * sizeof(Reduzido));

    int bytes = size * sizeof(Reduzido);
    Reduzido *d_odata = NULL, *d_idata = NULL;
	CHECK_ERROR(cudaMalloc((void **) &d_odata, numBlocks * sizeof(Reduzido))); // saída tem o tamanho do número de blocos
	CHECK_ERROR(cudaMalloc((void **) &d_idata, bytes));

    printf("%d blocks\n\n", numBlocks);


    Reduzido gpu_result = reducaoAuxiliar(size, tamanhoCaminho, tamanhoGrafo, custoOriginal, numThreads, numBlocks, maxThreads, maxBlocks,
                                    caminho_gpu, ads_gpu, custos_gpu, d_idata, d_odata, h_odata);

    printf("%d %d %.f\n", gpu_result.i, gpu_result.j, gpu_result.custo);

    free(h_odata);
    CHECK_ERROR(cudaFree(d_odata));
    CHECK_ERROR(cudaFree(d_idata));
}