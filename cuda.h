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
    *blocks = (n + threads - 1) / threads;

    if (*blocks > prop.maxGridSize[0]) {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        *blocks /= 2;
        *threads *= 2;
    }

}

__global__ void reduce2(int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, Reduzido* d_odata, unsigned int n) {
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : INFINITY;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            if (sdata[tid] > sdata[tid + s])
                sdata[tid] = sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void preparadorReducao(int size, int threads, int blocks, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, Reduzido* d_odata) {
    
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
	reduzir<<< dimGrid, dimBlock, smemSize >>>(caminho_gpu, ads_gpu, custos_gpu, d_odata, size);
}

Reduzido reducaoAuxiliar(int  n,
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  int  cpuFinalThreshold,
                  StopWatchInterface *timer,
                  int *caminho_gpu, ADS *ads_gpu, float *custos_gpu, Reduzido* d_idata, Reduzido* d_odata) {

    Reduzido gpu_result;
    gpu_result.i = - 1, gpu_result.j = -1, gpu_result.custo = INFINITY;
    bool needReadBack = true;

    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    // execute the kernel
    preparadorReducao(n, numThreads, numBlocks, whichKernel, caminho_gpu, ads_gpu, custos_gpu, d_odata);
    getLastCudaError("Kernel execution failed");

    cudaMemset(d_idata, 0, n * sizeof(Reduzido));

    // sum partial block sums on GPU
    int s=numBlocks;
    int kernel = whichKernel;

    while (s > cpuFinalThreshold) {
        
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, &blocks, &threads);
        cudaMemcpy(d_idata, d_odata, s * sizeof(Reduzido), cudaMemcpyDeviceToDevice);
        reduzir(s, threads, blocks, kernel, caminho_gpu, ads_gpu, custos_gpu, d_odata);

        s = (s + threads - 1) / threads;
    }

    if (s > 1) {
        // copy result from device to host
        checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));

        for (int i=0; i < s; i++) {
            if (gpu_result > h_odata[i])
                gpu_result = h_odata[i];
        }

        needReadBack = false;
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    if (needReadBack) {
        // copy final sum from device to host
        checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(Reduzido), cudaMemcpyDeviceToHost));
    }

    return gpu_result;
}

void runTest(int tamanhoCaminho, int *caminho_gpu, ADS *ads_gpu, float *custos_gpu) {
    
    int size = tamanhoCaminho * tamanhoCaminho;    // number of elements to reduce
    int maxThreads = 256;  // number of threads per block
    int maxBlocks = 64;
    int numBlocks = 0;
    int numThreads = 0;
    int cpuFinalThreshold = 1;

    printf("%d elements\n", size);
    printf("%d threads (max)\n", maxThreads);

    getNumBlocksAndThreads(size, maxBlocks, maxThreads, &numBlocks, &numThreads);

    if (numBlocks == 1) cpuFinalThreshold = 1;

    // allocate mem for the result on host side
    Reduzido *h_odata = (Reduzido *) malloc(numBlocks * sizeof(Reduzido));

    int bytes = size * sizeof(Reduzido);
    Reduzido *d_odata = NULL, *d_idata = NULL;
	checkCudaErrors(cudaMalloc((void **) &d_odata, numBlocks * sizeof(Reduzido))); // saída tem o tamanho do número de blocos
	checkCudaErrors(cudaMalloc((void **) &d_idata, bytes));

    printf("%d blocks\n\n", numBlocks);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);

    Reduzido gpu_result = reducaoAuxiliar(size, numThreads, numBlocks, maxThreads, maxBlocks,
                                    cpuFinalThreshold, timer, caminho_gpu, ads_gpu, custos_gpu, d_idata, d_odata);

    double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
    printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
           1.0e-9 * ((double)bytes)/reduceTime, reduceTime, size, 1, numThreads);

    free(h_odata);
    checkCudaErrors(cudaFree(d_odata));
}