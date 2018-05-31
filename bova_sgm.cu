#include "common.h"
#include "costs.h"
#include "util.h"
#include "aggregation.h"


static PixType *d_imgleft_data, *d_imgright_data;
static PixType *d_imgleft_grad, *d_imgright_grad; 
static PixType *d_clibTab;
static CostType *d_pixDiff;
static CostType *d_cost;
static CostType *d_hsum;
static CostType *d_sp;
static DispType *d_disp;
static CostType *d_mins;
static CostType *d_outdisp;

static DispType *d_disp2;
static CostType *d_disp2cost;

static DispType *h_disparity;


static int p1, p2;
static int blocksize;
static int rows, cols, img_size;
static int preFilterCap;
static int uniquenessRatio;
static int disp12MaxDiff;
const int TAB_OFS = 256*4, TAB_SIZE = 256 + TAB_OFS*2; 

void cuda_init(SGM_PARAMS *params)
{
	preFilterCap = params->preFilterCap; 
	p1 = params->P1;
	p2 = params->P2;
	blocksize = params->BlockSize;
	uniquenessRatio = params->uniquenessRatio;
	disp12MaxDiff = params->disp12MaxDiff;
}

cv::Mat compute_disparity(cv::Mat *left_img, cv::Mat *right_img, float *cost_time)
{
	if(CV_8UC1 !=  left_img->type() || CV_8UC1 != right_img->type())
	{
		std::cout<<"image type error\n";
		exit(-1);
	}
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
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_hsum, sizeof(CostType) * img_size * MAX_DISPARITY));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_sp, sizeof(CostType) * img_size * MAX_DISPARITY));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disp, sizeof(DispType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_mins, sizeof(CostType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_outdisp, sizeof(DispType) * img_size));	

		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disp2, sizeof(DispType) * img_size));	
		CUDA_CHECK_RETURN(cudaMalloc((void **)&d_disp2cost, sizeof(CostType) * img_size));	

		h_disparity = (DispType *)malloc(sizeof(DispType) * img_size);

		fill_tab<<<1, 1>>>(d_clibTab, TAB_SIZE, TAB_OFS, preFilterCap);
	}
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		std::cout<<"Error: "<<cudaGetErrorString(err)<<" "<<err<<std::endl;
		exit(-1);
	}
	
	double start = cv::getTickCount();
	CUDA_CHECK_RETURN(cudaMemcpy(d_imgleft_data, left_img->ptr<PixType>(), sizeof(PixType) * img_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_imgright_data, right_img->ptr<PixType>(), sizeof(PixType) * img_size, cudaMemcpyHostToDevice));
	
//	get_gradient<<<rows, WARP_SIZE>>>(d_imgleft_data, d_imgright_data, d_imgleft_grad, d_imgright_grad, d_clibTab + TAB_OFS, rows, cols);
	get_gradient<<<1, rows>>>(d_imgleft_data, d_imgright_data, d_imgleft_grad, d_imgright_grad, d_clibTab + TAB_OFS, rows, cols); //耗时，后期可以优化
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemset(d_pixDiff, 0, sizeof(CostType) * img_size * MAX_DISPARITY));
	get_pixel_diff<<<rows, MAX_DISPARITY>>>(d_imgleft_grad, d_imgright_grad, rows, cols, 0, d_pixDiff); 
	cudaDeviceSynchronize();
	get_pixel_diff<<<rows, MAX_DISPARITY>>>(d_imgleft_data, d_imgright_data, rows, cols, 2, d_pixDiff); 
	cudaDeviceSynchronize();

	get_hsum<<<rows, MAX_DISPARITY>>>(d_pixDiff, d_hsum, rows, cols, blocksize);
	cudaDeviceSynchronize();

	get_cost<<<cols - MAX_DISPARITY, MAX_DISPARITY>>>(d_hsum, d_cost, p2, rows, cols, blocksize);//d_cost前MAX_DISPARITY列没有用
	cudaDeviceSynchronize();
	
	CUDA_CHECK_RETURN(cudaMemset(d_sp, 0, sizeof(CostType) * img_size * MAX_DISPARITY));

	cost_aggregation_lr<<<rows, WARP_SIZE>>>(d_cost, d_sp, p1, p2, cols, rows);
	cudaDeviceSynchronize();

	cost_aggregation_ud_lr<<<cols - MAX_DISPARITY, WARP_SIZE>>>(d_cost, d_sp, p1, p2, cols, rows);
	err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		std::cout<<"Error: "<<cudaGetErrorString(err)<<" "<<err<<std::endl;
		exit(-1);
	}
	cudaDeviceSynchronize();

	cost_aggregation_ud<<<cols - MAX_DISPARITY, WARP_SIZE>>>(d_cost, d_sp, p1, p2, cols, rows);
	cudaDeviceSynchronize();

	cost_aggregation_rl_ud<<<cols - MAX_DISPARITY, WARP_SIZE>>>(d_cost, d_sp, p1, p2, cols, rows);
	cudaDeviceSynchronize();

	cost_aggregation_rl<<<rows, WARP_SIZE>>>(d_cost, d_sp, p1, p2, cols, rows);
	cudaDeviceSynchronize();

	get_disparity<<<rows, cols - MAX_DISPARITY>>>(d_sp, d_disp, d_mins, uniquenessRatio, d_disp2cost, d_disp2, cols, rows);
	cudaDeviceSynchronize();
	
	lrcheck<<<1, cols>>>(d_disp, d_mins, d_disp2, d_disp2cost, disp12MaxDiff, cols, rows);

	MedianFilter<<<(img_size + MAX_DISPARITY - 1)/MAX_DISPARITY, MAX_DISPARITY>>>(d_disp, d_outdisp, rows, cols);

	CUDA_CHECK_RETURN(cudaMemcpy(h_disparity, d_outdisp, sizeof(DispType) * img_size, cudaMemcpyDeviceToHost));

	double end = cv::getTickCount();
	printf("alogrithm cost:%fms\n", (end - start) * 1000 / cv::getTickFrequency());

#if 0
	
	double cpy_start = cv::getTickCount();
	PixType *h_grad = (PixType *)malloc(sizeof(PixType) * img_size);
	if(!h_grad)
	{
		printf("error\n");
	}
	CUDA_CHECK_RETURN(cudaMemcpy(h_grad, d_imgleft_grad, sizeof(PixType) * img_size, cudaMemcpyDeviceToHost));
	double cpy_end = cv::getTickCount();
	printf("copy data cost:%lfms\n", (cpy_end - cpy_start)*1000/cv::getTickFrequency());

	ofstream  gradient_file;
	gradient_file.open("grad_left.out", ios::out);
	for(int i= 0 ; i < rows; i++)
	for(int j = 0; j < cols; j++ )
			gradient_file<<"grad[row="<<i<<" col="<<j<<"]="<<(int)h_grad[i * cols +j]<<endl;
	gradient_file.close();

	CUDA_CHECK_RETURN(cudaMemcpy(h_grad, d_imgright_grad, sizeof(PixType) * img_size, cudaMemcpyDeviceToHost));
	gradient_file.open("grad_right.out", ios::out);
	for(int i= 0 ; i < rows; i++)
	for(int j = 0; j < cols; j++ )
			gradient_file<<"grad[row="<<i<<" col="<<j<<"]="<<(int)h_grad[i * cols +j]<<endl;
	gradient_file.close();

	CUDA_CHECK_RETURN(cudaMemcpy(h_grad, d_imgleft_data, sizeof(PixType) * img_size, cudaMemcpyDeviceToHost));
	gradient_file.open("date_left.out", ios::out);
	for(int i= 0 ; i < rows; i++)
	for(int j = 0; j < cols; j++ )
			gradient_file<<"data[row="<<i<<" col="<<j<<"]="<<(int)h_grad[i * cols +j]<<endl;
	gradient_file.close();

	CUDA_CHECK_RETURN(cudaMemcpy(h_grad, d_imgright_data, sizeof(PixType) * img_size, cudaMemcpyDeviceToHost));
	gradient_file.open("date_right.out", ios::out);
	for(int i= 0 ; i < rows; i++)
	for(int j = 0; j < cols; j++ )
			gradient_file<<"data[row="<<i<<" col="<<j<<"]="<<(int)h_grad[i * cols +j]<<endl;
	gradient_file.close();

	free(h_grad);
#endif


#if 0
	
	double cpy_start = cv::getTickCount();
	PixType *h_grad = (PixType *)malloc(sizeof(PixType) * img_size);
	if(!h_grad)
	{
		printf("error\n");
	}
	CUDA_CHECK_RETURN(cudaMemcpy(h_grad, d_imgleft_grad, sizeof(PixType) * img_size, cudaMemcpyDeviceToHost));
	double cpy_end = cv::getTickCount();
	printf("copy data cost:%lfms\n", (cpy_end - cpy_start)*1000/cv::getTickFrequency());

	ofstream  gradient_file;
	gradient_file.open("grad.out", ios::out);
	for(int i= 0 ; i < rows; i++)
	for(int j = 0; j < cols; j++ )
			gradient_file<<"grad[row="<<i<<" col="<<j<<"]="<<(int)h_grad[i * cols +j]<<endl;
	gradient_file.close();
	free(h_grad);
#endif

#if 0
	
	double cpy_start = cv::getTickCount();
	CostType *h_cost = (CostType *)malloc(sizeof(CostType) * img_size * MAX_DISPARITY);
	if(!h_cost)
	{
		printf("error\n");
	}
	CUDA_CHECK_RETURN(cudaMemcpy(h_cost, d_cost, sizeof(CostType) * img_size * MAX_DISPARITY, cudaMemcpyDeviceToHost));
	double cpy_end = cv::getTickCount();
	printf("copy data cost:%lfms\n", (cpy_end - cpy_start)*1000/cv::getTickFrequency());


	ofstream  cost0;
	cost0.open("cost.out", ios::out);
	for(int i=0;i<rows;i++)
	for(int j = MAX_DISPARITY; j < cols; j++ )
		for(int k=0; k < MAX_DISPARITY; k++)
			cost0<<"C[row="<<i<<" col="<<j<<" d="<<k<<"]: "<<h_cost[(i * cols + j)*MAX_DISPARITY + k]<<endl;
	cost0.close();
	free(h_cost);
#endif


#if 0
	
	double cpy_start = cv::getTickCount();
	CostType *h_sp = (CostType *)malloc(sizeof(CostType) * img_size * MAX_DISPARITY);
	if(!h_sp)
	{
		printf("error\n");
	}
	CUDA_CHECK_RETURN(cudaMemcpy(h_sp, d_sp, sizeof(CostType) * img_size * MAX_DISPARITY, cudaMemcpyDeviceToHost));
	double cpy_end = cv::getTickCount();
	printf("copy data cost:%lfms\n", (cpy_end - cpy_start)*1000/cv::getTickFrequency());


	ofstream  cost0;
	cost0.open("sp.out", ios::out);
	for(int i=0;i<rows;i++)
		for(int j = cols - 1; j >= MAX_DISPARITY; j-- )
			for(int k=0; k < MAX_DISPARITY; k++)
			cost0<<"y="<<i<<", x="<<j<<", d="<<k<<", sp="<<h_sp[(i * cols + j)*MAX_DISPARITY + k]<<endl;
	cost0.close();
	free(h_sp);
#endif

#if 1
	
	double cpy_start = cv::getTickCount();
	CostType *h_disp = (CostType *)malloc(sizeof(DispType) * img_size);
	if(!h_disp)
	{
		printf("error\n");
	}
	CUDA_CHECK_RETURN(cudaMemcpy(h_disp, d_disp2, sizeof(DispType) * img_size, cudaMemcpyDeviceToHost));
	double cpy_end = cv::getTickCount();
	printf("copy data cost:%lfms\n", (cpy_end - cpy_start)*1000/cv::getTickFrequency());


	ofstream  cost0;
	cost0.open("disp.out", ios::out);
	for(int i=0;i<rows;i++)
		for(int j = cols - 1; j >= MAX_DISPARITY; j-- )
			cost0<<"y="<<i<<", x="<<j<<", bestDisp="<<h_disp[i * cols + j]<<endl;
	cost0.close();
	free(h_disp);
#endif


	cv::Mat disparity_img(left_img->size(), CV_16S, h_disparity);	
	
	return disparity_img;
}


void free_gpu_mem()
{
	std::cout<<"Free Mem\n";
	CUDA_CHECK_RETURN(cudaFree(d_imgleft_data));
	d_imgleft_data = NULL;
	CUDA_CHECK_RETURN(cudaFree(d_imgright_data));
	d_imgright_data = NULL;
	CUDA_CHECK_RETURN(cudaFree(d_clibTab));
	d_clibTab = NULL;
	CUDA_CHECK_RETURN(cudaFree(d_pixDiff));
	d_pixDiff = NULL;
	CUDA_CHECK_RETURN(cudaFree(d_cost));
	d_cost = NULL;
	CUDA_CHECK_RETURN(cudaFree(d_hsum));
	d_hsum = NULL;
	CUDA_CHECK_RETURN(cudaFree(d_sp));
	d_sp = NULL;

	free(h_disparity);
	h_disparity = NULL;
}
