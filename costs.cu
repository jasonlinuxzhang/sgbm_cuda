#include "common.h"

__global__ void get_gradient(const PixType *d_imgleft_data, const PixType *d_imgright_data, PixType *d_imgleft_grad, PixType *d_imgright_grad, PixType *d_tab, int d_rows, int d_cols)
{
	int pixels_number_thread = d_cols / WARP_SIZE;
	int row = blockIdx.x;
	int col_start = threadIdx.x * pixels_number_thread;  //每个线程处理的范围的开始的横坐标
	int col = 0;
	int i = 0;
	int pre_row_add = 0, next_row_add = 0;
	
	if(col_start < d_cols)
	{
		if(0 == row)
		{
			pre_row_add = 0;
			next_row_add = d_cols;
		}
		else if(d_rows == row)
		{
			pre_row_add = 0 - d_cols;
			next_row_add = 0;
		}
		else
		{
			pre_row_add = 0 - d_cols;
			next_row_add = d_cols;
		}

		if(0 == col_start)  //改线程处理的是最左端的像素，包括横坐标为0
		{
			for(i = 1; i < pixels_number_thread; i++)
			{
				col = col_start + i;
				d_imgleft_grad[row * d_cols + col] = d_tab[ d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1] \
													    + d_imgleft_data[row * d_cols + col + 1 + pre_row_add] - d_imgleft_data[row * d_cols + col +1 + pre_row_add] \
													    + d_imgleft_data[row * d_cols + col + 1 + next_row_add] - d_imgleft_data[row * d_cols + col +1 + next_row_add] \
													   ];

				d_imgright_grad[row * d_cols + col] = d_tab[ d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1] \
													    + d_imgright_data[row * d_cols + col + 1 + pre_row_add] - d_imgright_data[row * d_cols + col +1 + pre_row_add] \
													    + d_imgright_data[row * d_cols + col + 1 + next_row_add] - d_imgright_data[row * d_cols + col +1 + next_row_add] \
													   ];
			}
			d_imgleft_grad[row * d_cols + 0] = d_tab[0];
			d_imgright_grad[row * d_cols + 0] = d_tab[0];
		}
		else if(col_start + pixels_number_thread > d_cols)  //处理最右端的像素,因为这一个线程可能和其他线程处理的像素个数不一样
		{
			int loop_end = d_cols % WARP_SIZE; 
			for(i = 0; i < loop_end; i++)
			{
				col = col_start + i;
				d_imgleft_grad[row * d_cols + col] = d_tab[ d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1] \
													    + d_imgleft_data[row * d_cols + col + 1 + pre_row_add] - d_imgleft_data[row * d_cols + col +1 + pre_row_add] \
													    + d_imgleft_data[row * d_cols + col + 1 + next_row_add] - d_imgleft_data[row * d_cols + col +1 + next_row_add] \
													   ];

				d_imgright_grad[row * d_cols + col] = d_tab[ d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1] \
													    + d_imgright_data[row * d_cols + col + 1 + pre_row_add] - d_imgright_data[row * d_cols + col +1 + pre_row_add] \
													    + d_imgright_data[row * d_cols + col + 1 + next_row_add] - d_imgright_data[row * d_cols + col +1 + next_row_add] \
													   ];
			
			}

		}
		else
		{
			for(i = 0; i < pixels_number_thread; i++)
			{
				col = col_start + i;
				d_imgleft_grad[row * d_cols + col] = d_tab[ d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1] \
														    + d_imgleft_data[row * d_cols + col + 1 + pre_row_add] - d_imgleft_data[row * d_cols + col +1 + pre_row_add] \
														    + d_imgleft_data[row * d_cols + col + 1 + next_row_add] - d_imgleft_data[row * d_cols + col +1 + next_row_add] \
														   ];

				d_imgright_grad[row * d_cols + col] = d_tab[ d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1] \
														    + d_imgright_data[row * d_cols + col + 1 + pre_row_add] - d_imgright_data[row * d_cols + col +1 + pre_row_add] \
														    + d_imgright_data[row * d_cols + col + 1 + next_row_add] - d_imgright_data[row * d_cols + col +1 + next_row_add] \
														   ];
			}
		}

	}

}


//每行有MAX_DISPARITY个线程，每个线程处理该行所有点的特定视差
__global__ void get_pixel_diff(const PixType * d_imgleft_buf, const PixType * d_imgright_buf, int rows, int cols, int diff_scale, CostType *d_cost) 
{
	int row = blockIdx.x;
	int now_disparity = threadIdx.x;

	const PixType *local_imgleft_buf = d_imgleft_buf + row * cols;
	const PixType *local_imgright_buf = d_imgleft_buf + row * cols;
	
	for(int x = 0; x < cols; x++)
	{
		int u = local_imgleft_buf[x];
		int ul = x > 0 ? (u + local_imgleft_buf[x - 1])/2 : u;
		int ur = x < cols - 1 ? (u + local_imgleft_buf[x + 1]/2) : u;	

		int u0 = min(ul, ur); u0 = min(u0, u);
		int u1 = max(ul, ur); u1 = max(u1, u); 


		int v = x >= now_disparity ? local_imgright_buf[x - now_disparity] : 0;
		int vl = x >= now_disparity + 1 ? local_imgright_buf[x - now_disparity - 1] : 0;
		int vr = x >= now_disparity - 1 ? local_imgright_buf[x - now_disparity + 1] : 0;

		int v0 = min(vl, vr); v0 = min(v0, v);
		int v1 = max(vl, vr); v1= max(v1, v);

		int c0 = max(0, u - v1); c0 = max(c0, v0 - u);
		int c1 = max(0, v - u1); c1 = max(c1, u0 - v);

		d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity] = d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity] + min(c0, c1) >> diff_scale;
	}

}


__global__ void get_hsum(const CostType *d_pixel_diff, CostType *d_hsumAdd, int rows, int cols)
{
	int row = blockIdx.x;
	int now_disparity = threadIdx.x;

	CostType *local_hsumAdd = d_hsumAdd + row * cols * MAX_DISPARITY;
	const CostType *local_pixel_diff = d_pixel_diff + row * cols * MAX_DISPARITY;
	
	for(int x = 0; x < SW2 * MAX_DISPARITY; x += MAX_DISPARITY)
	{
		int scale = x == 0 ? SW2 + 1 : 1;
		local_hsumAdd[now_disparity] = local_hsumAdd[now_disparity] + local_pixel_diff[x + now_disparity] * scale;
	}

	for(int x = MAX_DISPARITY; x < cols * MAX_DISPARITY; x += MAX_DISPARITY)
	{
		const CostType *pixAdd = local_pixel_diff + min(x + SW2 * MAX_DISPARITY, (cols - 1) * MAX_DISPARITY);
		const CostType *pixSub = local_pixel_diff + max(x - (SW2 + 1) * MAX_DISPARITY, 0);
		local_hsumAdd[x + now_disparity] = local_hsumAdd[x - MAX_DISPARITY + now_disparity] + pixAdd[now_disparity] - pixSub[now_disparity];
	}
}

__global__ void get_cost(const CostType *d_hsumAdd, CostType *d_cost, int p2, int rows, int cols)
{
	int col = blockIdx.x; //线程块代表每一列
	int now_disparity = threadIdx.x; //线程块中的一个线程处理某一列的视差为d时的代价
	
	CostType *local_cost = d_cost + col * MAX_DISPARITY;
	const CostType *local_hsumAdd = d_hsumAdd + col * MAX_DISPARITY;
	
	//y == 0
	local_cost[0 + now_disparity] = p2 + local_hsumAdd[0 + now_disparity] * (SH2 + 1);
	local_cost[0 + now_disparity] = local_cost[0 + now_disparity] + local_hsumAdd[cols * MAX_DISPARITY + now_disparity]; 
	local_cost[0 + now_disparity] = local_cost[0 + now_disparity] + local_hsumAdd[cols * MAX_DISPARITY * 2 + now_disparity]; 
 
	for(int y = 1; y < rows; y++)
	{
		const CostType *h_sumSub = local_hsumAdd+ (y >= 5 ? cols * MAX_DISPARITY * (y - 5) : 0);
		
		local_cost[cols * MAX_DISPARITY * y + now_disparity] = local_cost[cols * MAX_DISPARITY * (y - 1)] + local_hsumAdd[cols * MAX_DISPARITY * y + now_disparity] - h_sumSub[now_disparity];
	}
}

__global__ void fill_tab(PixType *d_tab, int TAB_SIZE, int TAB_OFS, int ftzero)
{
	for(int k = 0; k < TAB_SIZE; k++)
	{
		d_tab[k] = (PixType)(min(max(k - TAB_OFS, -ftzero), ftzero) + ftzero); 
	}
}
