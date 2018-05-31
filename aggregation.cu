#include "aggregation.h"
#include "util.h"

__host__ __device__ CostType safe_add(int a, int b )
{
	int t = a + b;
	if(32767 < t)
		return 32767;
	return t;
}

/*
	<<<(rows + WARP_SIZE)/WARP_SIZE, WARP_SIZE>>>
	sp 需要初始化为非法值
*/
/*
__global__ void cost_aggregation_lr(const CostType *d_cost, CostType *sp, int p1, int p2, int cols, int rows)
{	
	int row  = blockIdx.x * WARP_SIZE + threadIdx.x;
	if(row < rows)
	{
		const CostType *local_cost = d_cost + row * cols * MAX_DISPARITY;
		CostType *local_sp = sp + row * cols * MAX_DISPARITY;

		int _lr_pre[MAX_DISPARITY + 2] = {0}; //用于保存前面一个点的lr,按照opencv，多两个是为了处理-1和D
		int _lr_pre_temp[MAX_DISPARITY + 2] = {0}; 
		int delta = 0; //前一点的，lr的最小值（d在0--D-1中）

		int *lr_pre = _lr_pre + 1;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = SHRT_MAX;	
		
		int *lr_pre_temp = _lr_pre_temp + 1;
		lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;	
		
		int minlr = SHRT_MAX;
		for(int x = MAX_DISPARITY; x < cols; x++)
		{
			local_cost = local_cost + x * MAX_DISPARITY;
			local_sp  = local_sp + x * MAX_DISPARITY;
			for(int d = 0; d < MAX_DISPARITY; d++)
			{
				lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
				
				minlr = min(minlr, lr_pre_temp[d]);	
				local_sp[d] = safe_add((int)local_cost[d], lr_pre_temp[d]);
			}		
			delta = minlr;
			int * p_t = lr_pre_temp;
			lr_pre_temp = lr_pre;
			lr_pre = p_t;
		}
	}
}

*/

/*
	rows个线程块，一个线程块包含warp（32）个线程，一个线程块处理1行，每个线程处理该行的4个视差
*/
__global__ void cost_aggregation_lr(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows)
{
	int row = blockIdx.x;
	int disparity_start = threadIdx.x * (MAX_DISPARITY / WARP_SIZE);
	const CostType *local_cost = d_cost + (row * cols + MAX_DISPARITY)* MAX_DISPARITY; //属于该行的起始地址, 每行的前MAX_DISPARITY列不能计算
	CostType *local_sp = d_sp + (row * cols + MAX_DISPARITY) * MAX_DISPARITY;
	

	__shared__ int delta;
	__shared__ int _lr_pre[MAX_DISPARITY + 2];
	__shared__ int _lr_pre_temp[MAX_DISPARITY + 2];
	int *lr_pre = _lr_pre + 1;
	int *lr_pre_temp = _lr_pre_temp + 1;

	if(0 == threadIdx.x)
	{
		delta = 0 + p2;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
	}
	//利用各个线程，设置对应的lr_pre初始值
	lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

	__syncthreads();
	
	
	
    lr_pre[-1] = lr_pre[MAX_DISPARITY] = SHRT_MAX;  
    lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX; 	

	for(int col = 0; col < cols- MAX_DISPARITY; col++)
	{
		int minlr = SHRT_MAX;  
		int d = disparity_start;		
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/
		
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/		

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/
		
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		int min_pre_lr = warpReduceMin(minlr);
		
		if(0 == threadIdx.x)
			delta = min_pre_lr + p2;
		
		int * pt = lr_pre;
		lr_pre = lr_pre_temp;
		lr_pre_temp = pt;	

		local_cost = local_cost + MAX_DISPARITY;
		local_sp  = local_sp + MAX_DISPARITY;
	}
}

/*
	上-下 & 左-右, 上半部分
	每个线程块处理一个上斜线
*/
__global__ void cost_aggregation_ud_lr(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows)
{
	int col = blockIdx.x;
	int row = 0;
	int disparity_start = threadIdx.x * (MAX_DISPARITY / WARP_SIZE);

	const CostType *local_cost = d_cost + (MAX_DISPARITY + col)* MAX_DISPARITY; //属于该行的起始地址, 每行的前MAX_DISPARITY列不能计算
	CostType *local_sp = d_sp + (MAX_DISPARITY + col) * MAX_DISPARITY;
	
	__shared__ int delta;
	__shared__ int _lr_pre[MAX_DISPARITY + 2];
	__shared__ int _lr_pre_temp[MAX_DISPARITY + 2];
	int *lr_pre = _lr_pre + 1;
	int *lr_pre_temp = _lr_pre_temp + 1;

	if(0 == threadIdx.x)
	{
		delta = 0 + p2;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
	}
	lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

	__syncthreads();
	
	
    lr_pre[-1] = lr_pre[MAX_DISPARITY] = SHRT_MAX; //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
    lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX; 	

	for( ; (col < cols - MAX_DISPARITY) && (row < rows); col++, row++)
	{
		int minlr = SHRT_MAX;  
		int d = disparity_start;		
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/
		
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		int min_pre_lr = warpReduceMin(minlr);
		
		if(0 == threadIdx.x)
			delta = min_pre_lr + p2;
		
		int * pt = lr_pre;
		lr_pre = lr_pre_temp;
		lr_pre_temp = pt;	

		local_cost = local_cost + (cols + 1) * MAX_DISPARITY;
		local_sp  = local_sp + (cols + 1) * MAX_DISPARITY;
	}

	__syncthreads();	

	
	//下部分进行计算, 经过上面的循环，col或者row超界了
	if((col == cols - MAX_DISPARITY) && (row != rows)) 
	{
		col = 0;
		//printf("row=%d, col=%d\n", row, col);

		if(0 == threadIdx.x)
		{
			delta = 0 + p2;
			lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
		}
		lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

		__syncthreads();

		lr_pre = _lr_pre + 1;
		lr_pre_temp = _lr_pre_temp + 1;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = SHRT_MAX;  
		lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX; 	

		local_cost = d_cost + (row * cols + MAX_DISPARITY)* MAX_DISPARITY; //属于该行的起始地址, 每行的前MAX_DISPARITY列不能计算
		local_sp = d_sp + (row * cols + MAX_DISPARITY) * MAX_DISPARITY;

		for( ; col < cols - MAX_DISPARITY && row < rows; col++, row++)
		{
			int minlr = SHRT_MAX;  
			int d = disparity_start;		
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			/*(
			   if(row == 0 && col == 0)
			   {
			   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
			   }
			*/

			d++;
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			/*
			   if(row == 1 && col == 0)
			   {
			   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
			   }
			*/

			d++;
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			/*
			   if(row == 1 && col == 0)
			   {
			   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
			   }
			*/

			d++;
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			/*
			   if(row == 1 && col == 0)
			   {
			   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
			   }
			*/

			int min_pre_lr = warpReduceMin(minlr);

			if(0 == threadIdx.x)
				delta = min_pre_lr + p2;

			int * pt = lr_pre;
			lr_pre = lr_pre_temp;
			lr_pre_temp = pt;	

			local_cost = local_cost + (cols + 1) * MAX_DISPARITY;
			local_sp  = local_sp + (cols + 1) * MAX_DISPARITY;
		}
	}

}

__global__ void cost_aggregation_ud(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows)
{
	int col = blockIdx.x;
	int disparity_start = threadIdx.x * (MAX_DISPARITY / WARP_SIZE);
	const CostType *local_cost = d_cost + (col + MAX_DISPARITY)* MAX_DISPARITY; //属于该列的起始地址, 每行的前MAX_DISPARITY列不能计算
	CostType *local_sp = d_sp + (col + MAX_DISPARITY) * MAX_DISPARITY;
	

	__shared__ int delta;
	__shared__ int _lr_pre[MAX_DISPARITY + 2];
	__shared__ int _lr_pre_temp[MAX_DISPARITY + 2];
	int *lr_pre = _lr_pre + 1;
	int *lr_pre_temp = _lr_pre_temp + 1;

	if(0 == threadIdx.x)
	{
		delta = 0 + p2;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
	}
	//利用各个线程，设置对应的lr_pre初始值
	lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

	__syncthreads();
	

	for(int row = 0; row < rows; row++)
	{
		int minlr = SHRT_MAX;  
		int d = disparity_start;		
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/	
	
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/		

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0 && col == 1)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		int min_pre_lr = warpReduceMin(minlr);
		
		if(0 == threadIdx.x)
			delta = min_pre_lr + p2;
		
		int * pt = lr_pre;
		lr_pre = lr_pre_temp;
		lr_pre_temp = pt;	

		local_cost = local_cost + cols * MAX_DISPARITY;
		local_sp  = local_sp + cols * MAX_DISPARITY;
	}
}


__global__ void cost_aggregation_rl_ud(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows)
{
	int col = blockIdx.x; //共cols - MAX_DISPARITY个线程块
	int row = 0;
	int disparity_start = threadIdx.x * (MAX_DISPARITY / WARP_SIZE);

	const CostType *local_cost = d_cost + (MAX_DISPARITY + col)* MAX_DISPARITY; //属于该行的起始地址, 每行的前MAX_DISPARITY列不能计算
	CostType *local_sp = d_sp + (MAX_DISPARITY + col) * MAX_DISPARITY;
	
	__shared__ int delta;
	__shared__ int _lr_pre[MAX_DISPARITY + 2];
	__shared__ int _lr_pre_temp[MAX_DISPARITY + 2];
	int *lr_pre = _lr_pre + 1;
	int *lr_pre_temp = _lr_pre_temp + 1;

	if(0 == threadIdx.x)
	{
		delta = 0 + p2;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
	}
	lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

	__syncthreads();
	
	for( ; (col >= 0) && (row < rows); col--, row++)
	{
		int minlr = SHRT_MAX;  
		int d = disparity_start;		
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row < 2)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row < 2)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row < 2)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/	
	
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row < 2)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		int min_pre_lr = warpReduceMin(minlr);
		
		if(0 == threadIdx.x)
			delta = min_pre_lr + p2;
		
		int * pt = lr_pre;
		lr_pre = lr_pre_temp;
		lr_pre_temp = pt;	

		local_cost = local_cost + (cols - 1) * MAX_DISPARITY;
		local_sp  = local_sp + (cols - 1) * MAX_DISPARITY;
	}

	__syncthreads();	

	
	//下部分进行计算, 经过上面的循环，col或者row超界了
	if((col == -1) && (row != rows)) 
	{
		col = cols - MAX_DISPARITY - 1;

		if(0 == threadIdx.x)
		{
			delta = 0 + p2;
			lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
		}
		lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

		__syncthreads();

		lr_pre = _lr_pre + 1;
		lr_pre_temp = _lr_pre_temp + 1;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = SHRT_MAX;  
		lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX; 	

		local_cost = d_cost + (row * cols + col +  MAX_DISPARITY)* MAX_DISPARITY; //属于该行的起始地址, 每行的前MAX_DISPARITY列不能计算
		local_sp = d_sp + (row * cols + col + MAX_DISPARITY) * MAX_DISPARITY;

		for( ; (col >= 0) && (row < rows); col--, row++)
		{
			int minlr = SHRT_MAX;  
			int d = disparity_start;		
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			//if(row == 2)
			//   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);

			d++;
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			//if(row == 2)
			//   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);

			d++;
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			//if(row == 2)
			//   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);

			d++;
			lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
			minlr = min(minlr, lr_pre_temp[d]);
			local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
			//if(row == 2)
			//   printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);

			int min_pre_lr = warpReduceMin(minlr);

			if(0 == threadIdx.x)
				delta = min_pre_lr + p2;

			int * pt = lr_pre;
			lr_pre = lr_pre_temp;
			lr_pre_temp = pt;	

			local_cost = local_cost + (cols - 1) * MAX_DISPARITY;
			local_sp  = local_sp + (cols - 1) * MAX_DISPARITY;
		}
	}

}


__global__ void cost_aggregation_rl(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows)
{
	int row = blockIdx.x;
	int disparity_start = threadIdx.x * (MAX_DISPARITY / WARP_SIZE);
	const CostType *local_cost = d_cost + (row * cols + cols - 1)* MAX_DISPARITY; //属于该行最右边一列起始地址, 每行的前MAX_DISPARITY列不能计算
	CostType *local_sp = d_sp + (row * cols + cols - 1) * MAX_DISPARITY;
	

	__shared__ int delta;
	__shared__ int _lr_pre[MAX_DISPARITY + 2];
	__shared__ int _lr_pre_temp[MAX_DISPARITY + 2];
	int *lr_pre = _lr_pre + 1;
	int *lr_pre_temp = _lr_pre_temp + 1;

	if(0 == threadIdx.x)
	{
		delta = 0 + p2;
		lr_pre[-1] = lr_pre[MAX_DISPARITY] = lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX;
	}
	//利用各个线程，设置对应的lr_pre初始值
	lr_pre[disparity_start] = lr_pre[disparity_start + 1] = lr_pre[disparity_start + 2] = lr_pre[disparity_start + 3] = lr_pre_temp[disparity_start] = lr_pre_temp[disparity_start + 1] = lr_pre_temp[disparity_start + 2] = lr_pre_temp[disparity_start + 3] = 0;

	__syncthreads();
	
	
    lr_pre[-1] = lr_pre[MAX_DISPARITY] = SHRT_MAX;  
    lr_pre_temp[-1] = lr_pre_temp[MAX_DISPARITY] = SHRT_MAX; 	

	for(int col = cols - MAX_DISPARITY -1; col >= 0; col--)
	{
		int minlr = SHRT_MAX;  
		int d = disparity_start;		
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/
		
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/
		
		d++;
		lr_pre_temp[d] = local_cost[d] +  min(lr_pre[d], min(lr_pre[d - 1] + p1, min(lr_pre[d+1] + p1, delta))) - delta;
		minlr = min(minlr, lr_pre_temp[d]);
		local_sp[d] = safe_add(local_sp[d], lr_pre_temp[d]);
		/*
		if(row == 0)
		{
			printf("y=%d, x=%d, d=%d, l=%d, cost=%d, delta=%d\n", row, col + MAX_DISPARITY, d, lr_pre_temp[d], local_cost[d], delta);
		}
		*/

		int min_pre_lr = warpReduceMin(minlr);
		
		if(0 == threadIdx.x)
			delta = min_pre_lr + p2;
		
		int * pt = lr_pre;
		lr_pre = lr_pre_temp;
		lr_pre_temp = pt;	

		local_cost = local_cost - MAX_DISPARITY;
		local_sp  = local_sp - MAX_DISPARITY;
	}
}



__global__ void get_disparity(const CostType *d_sp, DispType *d_disp, CostType *d_mins, int uniquenessRatio, CostType *disp2cost, DispType *disp2, int cols, int rows)
{
	int row = blockIdx.x;
	int col = threadIdx.x + MAX_DISPARITY;

	int minS = SHRT_MAX;
	int bestDisp = -1;
	const CostType *local_sp = d_sp + (row * cols + col) * MAX_DISPARITY;
	int d = 0;

	d_disp[row * cols + col] = INVALID_DISP_SCALED;
	d_mins[row * cols + col] = SHRT_MAX;

	for(d = 0; d < MAX_DISPARITY; d++)
	{
		if(minS > local_sp[d])
		{
			minS = local_sp[d];
			bestDisp = d;
		}
	}

	for(d = 0; d < MAX_DISPARITY; d++)
	{
		if(local_sp[d] * (100 - uniquenessRatio) < minS * 100 && abs(bestDisp -d) > 1)
		{
			break;
		}
	}
	
	if(d < MAX_DISPARITY)  //说明求得的视差不对，该点视差值取-16
		return;	

	if(0 < bestDisp && bestDisp < MAX_DISPARITY - 1)
	{
		int denom2 = max(local_sp[bestDisp - 1] + local_sp[bestDisp + 1] - 2*local_sp[bestDisp], 1);
		bestDisp = bestDisp * DISP_SCALE + ((local_sp[bestDisp - 1] - local_sp[bestDisp + 1]) * DISP_SCALE + denom2)/(denom2*2);
	}
	else
		bestDisp = bestDisp * DISP_SCALE;

//	printf("y=%d, x=%d, bestDisp=%d\n", row, col, bestDisp);
	d_disp[row * cols + col] = bestDisp;

	d_mins[row * cols + col] = minS;
}

__global__ void lrcheck(DispType * d_disp, const CostType * d_mins, DispType *disp2, CostType *disp2cost, int disp12MaxDiff, int cols, int rows)
{
	int row = threadIdx.x;
	CostType *local_disp2cost = disp2cost + row * cols;
	DispType *local_disp2 = disp2 + row * cols;
	const CostType *local_dmins = d_mins + row * cols;
	DispType *local_disp = d_disp + row * cols;

	for(int col = MAX_DISPARITY; col < cols; col++)
	{
		local_disp2cost[col] = SHRT_MAX;
		local_disp2[col] = INVALID_DISP_SCALED;
	}	

	for(int col = MAX_DISPARITY; col < cols; col++)
	{
		int d = local_disp[col];
		int _x2 = col - d;
		if(local_disp2cost[_x2] > local_disp[col])
		{
			local_disp2cost[_x2] = local_dmins[col];
			local_disp2[_x2] = d;	
		}	
	}

	for(int col = MAX_DISPARITY; col < cols; col++)
	{
		int d1 = local_disp[col];
		if(d1 == INVALID_DISP_SCALED)
		{
			continue;
		}

		int _d = d1 >> DISP_SHIFT;
		int d_ = (d1 + DISP_SCALE - 1) >> DISP_SHIFT;
		int _x = col - _d, x_ = col - d_;
		if(0 <= _x && _x < cols && local_disp2[_x] >= 0 && abs(local_disp2[_x] - _d) > disp12MaxDiff && 0 <=x_ && x_ < cols && local_disp2[x_] >= 0 && abs(local_disp2[x_] - d_) > disp12MaxDiff)
			local_disp[col] = INVALID_DISP_SCALED;
	}
}

__global__ void MedianFilter(const DispType *d_input, DispType *d_out, int rows, int cols) 
{
    const uint32_t idx = blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t row = idx / cols;
    const uint32_t col = idx % cols;
	const int n = 3;
    DispType window[n*n];
    int half = n/2;

    if(row >= half && col >= half && row < rows-half && col < cols-half) {
        for(uint32_t i = 0; i < n; i++) {
            for(uint32_t j = 0; j < n; j++) {
                window[i*n+j] = d_input[(row-half+i)*cols+col-half+j];
            }
        }

        for(uint32_t i = 0; i < (n*n/2)+1; i++) {
            uint32_t min_idx = i;
            for(uint32_t j = i+1; j < n*n; j++) {
                if(window[j] < window[min_idx]) {
                    min_idx = j;
                }
            }
            const DispType tmp = window[i];
            window[i] = window[min_idx];
            window[min_idx] = tmp;
        }
        d_out[idx] = window[n*n/2];
    } else if(row < rows && col < cols) {
        d_out[idx] = d_input[idx];
    }
}

