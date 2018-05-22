#include "common.h"
#include "costs.h"
#include "util.h"


static PixType *d_imgleft_data, *d_imgright_data;
static PixType *d_imgleft_grad, *d_imgright_grad; 
static PixType *d_clibTab;
static CostType *d_pixDiff;
static CostType *d_cost;
static CostType *d_hsumAdd;


static int p1, p2;
static int rows, cols, img_size;
static int preFilterCap;
const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2; 

void cuda_init(SGM_PARAMS *params)
{
	preFilterCap = params->preFilterCap; 
	p1 = params->P1;
	p2 = params->P2;
}

cv::Mat compute_disparity(cv::Mat *left_img, cv::Mat *right_img, float *cost_time)
{
	assert(CV_8UC1 !=  left_img->type() && CV_8UC1 != right_img->type());
	static bool is_first_called = true;
	if(is_first_called)
	{
		std::cout<<"First Called\n";
		is_first_called = false;
		rows = left_img->rows;
		cols = left_img->cols;
		img_size = rows * cols;
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_imgleft_data, sizeof(PixType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_imgright_data, sizeof(PixType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_imgleft_grad, sizeof(PixType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_imgright_grad, sizeof(PixType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_clibTab, sizeof(PixType) * TAB_SIZE));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_pixDiff, sizeof(CostType) * img_size * MAX_DISPARITY));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_cost, sizeof(CostType) * img_size * MAX_DISPARITY));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_hsumAdd, sizeof(CostType) * img_size * MAX_DISPARITY));	


		fill_tab<<<1, 0>>>(d_clibTab, TAB_SIZE, TAB_OFS, preFilterCap);
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_imgleft_data, left_img->ptr<PixType>(), sizeof(PixType) * img_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyAsync(d_imgright_data, right_img->ptr<PixType>(), sizeof(PixType) * img_size, cudaMemcpyHostToDevice));
	
	get_gradient<<<rows, WARP_SIZE>>>(d_imgleft_data, d_imgright_data, d_imgleft_grad, d_imgright_grad, d_clibTab + TAB_OFS, rows, cols);

	get_pixel_diff<<<rows, MAX_DISPARITY>>>(d_imgleft_grad, d_imgright_grad, rows, cols, 0, d_pixDiff); 
	get_pixel_diff<<<rows, MAX_DISPARITY>>>(d_imgleft_data, d_imgright_data, rows, cols, 2, d_pixDiff); 

	
	get_hsum<<<rows, MAX_DISPARITY>>>(d_pixDiff, d_hsumAdd, rows, cols);

	get_cost<<<cols, MAX_DISPARITY>>>(d_hsumAdd, d_cost, p2, rows, cols);


	CostType *h_cost = (CostType *)malloc(sizeof(CostType) * img_size * MAX_DISPARITY);
	if(!h_cost)
	{
		printf("error\n");
	}
	CUDA_CHECK_RETURN(cudaMemcpyAsync(h_cost, d_cost, sizeof(CostType) * img_size * MAX_DISPARITY, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0);

	cudaDeviceSynchronize();
	float cost_t = 0;
	cudaEventElapsedTime(&cost_t, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	ofstream  cost0;
	cost0.open("cost.out", ios::in);
	for(int i=0;i<rows;i++)
	for(int j = 0; j < cols; j++ )
		for(int k=0; k < MAX_DISPARITY; k++)
			cost0<<"C[row="<<i<<" col="<<j<<" d="<<k<<"]: "<<h_cost[(i * cols + j)*MAX_DISPARITY + k]<<endl;
	cost0.close();


	printf("%f\n", cost_t);
	
	return *left_img;
}


void free_gpu_mem()
{
	std::cout<<"Free Mem\n";
	CUDA_CHECK_RETURN(cudaFree(d_imgleft_data));
	CUDA_CHECK_RETURN(cudaFree(d_imgright_data));
	CUDA_CHECK_RETURN(cudaFree(d_clibTab));
}
