#define CHECK_ERROR(call) do {                                                    \
	if(cudaSuccess != call) {                                                     \
		fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", cudaGetErrorString(call),  __FILE__, __LINE__); \
		exit(0);                                                                                 \
    } } while (0)


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