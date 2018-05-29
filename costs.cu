#include "common.h"

/*
__global__ void get_gradient(PixType *d_imgleft_data, PixType *d_imgright_data, PixType *d_imgleft_grad, PixType *d_imgright_grad, PixType *d_tab, int d_rows, int d_cols)
{
	int pixels_number_thread = d_cols / WARP_SIZE;
	int row = blockIdx.x;  //该线程所在行
	int col_start = threadIdx.x * pixels_number_thread;  //每个线程处理的范围的开始的横坐标
	int col = 0;
	int i = 0;
	int pre_row_add = 0, next_row_add = 0;
	
	if(col_start < d_cols)
	{
		if(0 == row)//第一行
		{
			pre_row_add = 0;
			next_row_add = 1;
		}
		else if(d_rows -1 == row) //最后一行
		{
			pre_row_add = 1;
			next_row_add = 0;
		}
		else
		{
			pre_row_add = 1;
			next_row_add = 1;
		}

		if(0 == col_start)  //改线程处理的是最左端的像素，包括横坐标为0
		{
			for(i = 1; i < pixels_number_thread; i++)
			{
				col = col_start + i;
				d_imgleft_grad[row * d_cols + col] = d_tab[ (d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1]) * 2 \
													    + d_imgleft_data[(row - pre_row_add) * d_cols + col + 1] - d_imgleft_data[(row - pre_row_add) * d_cols + col - 1] \
													    + d_imgleft_data[(row + next_row_add) * d_cols + col + 1] - d_imgleft_data[(row + next_row_add) * d_cols + col - 1] \
													   ];
				d_imgright_grad[row * d_cols + col] = d_tab[ (d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1]) * 2 \
													    + d_imgright_data[(row - pre_row_add) * d_cols + col + 1] - d_imgright_data[(row - pre_row_add) * d_cols + col - 1] \
													    + d_imgright_data[(row + next_row_add) * d_cols + col + 1] - d_imgright_data[(row + next_row_add) * d_cols + col - 1] \
													   ];
				//printf("row=%d, x=%d, right_data=%d\n", row, col, d_imgright_data[row * d_cols + col]);

	//			printf("row=%d,col=%d,grad=%d, pre_row=%d, org_imgdata=%d\n", row, col, d_imgleft_grad[row * d_cols + col], (row - pre_row_add), d_imgleft_data[row * d_cols + col ]);
				//printf("row=%d,col=%d,grad=%d, pre_row=%d, org_imgdata=%d\n", row, col, d_imgright_grad[row * d_cols + col], (row - pre_row_add), d_imgright_data[row * d_cols + col ]);
				//printf("y=%d,%d=%d\n", row, col,  d_imgright_grad[row * d_cols + col]);
			}
			d_imgleft_grad[row * d_cols + 0] = d_tab[0];
			d_imgright_grad[row * d_cols + 0] = d_tab[0];


			d_imgleft_data[row * d_cols + 0] = d_tab[0];
			d_imgright_data[row * d_cols + 0] = d_tab[0]; //此处仿照opencv代码，感觉opencv不对

		}
		else if(col_start + pixels_number_thread > d_cols - 1)  //处理最右端的像素,因为这一个线程可能和其他线程处理的像素个数不一样
		{
			for(i = 0; col_start + i < d_cols; i++)
			{
				col = col_start + i;
				d_imgleft_grad[row * d_cols + col] = d_tab[ (d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1]) * 2 \
													    + d_imgleft_data[(row - pre_row_add) * d_cols + col + 1] - d_imgleft_data[(row - pre_row_add) * d_cols + col - 1] \
													    + d_imgleft_data[(row + next_row_add) * d_cols + col + 1] - d_imgleft_data[(row + next_row_add) * d_cols + col - 1] \
													   ];
				d_imgright_grad[row * d_cols + col] = d_tab[ (d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1]) * 2 \
													    + d_imgright_data[(row - pre_row_add) * d_cols + col + 1] - d_imgright_data[(row - pre_row_add) * d_cols + col - 1] \
													    + d_imgright_data[(row + next_row_add) * d_cols + col + 1] - d_imgright_data[(row + next_row_add) * d_cols + col - 1] \
													   ];
			
				//printf("row=%d, x=%d, right_data=%d\n", row, col, d_imgright_data[row * d_cols + col]);
			//	printf("row=%d,col=%d,grad=%d, pre_row=%d, org_imgdata=%d\n", row, col, d_imgleft_grad[row * d_cols + col], (row - pre_row_add), d_imgleft_data[row * d_cols + col ]);
				//printf("row=%d,col=%d,grad=%d, pre_row=%d, org_imgdata=%d\n", row, col, d_imgright_grad[row * d_cols + col], (row - pre_row_add), d_imgright_data[row * d_cols + col ]);
				//printf("y=%d,%d=%d\n", row, col,  d_imgright_grad[row * d_cols + col]);
			}
			d_imgleft_grad[row * d_cols + d_cols - 1] = d_tab[0];
			d_imgright_grad[row * d_cols + d_cols - 1] = d_tab[0];
		//	printf("row=%d, col=639 grad=%d\n", row, d_tab[0]);

			d_imgleft_data[row * d_cols + d_cols - 1] = d_tab[0];
			d_imgright_data[row * d_cols + d_cols - 1] = d_tab[0]; //此处仿照opencv代码，感觉opencv不对
				//printf("row=%d, x=%d, right_data=%d\n", row, d_cols-1, d_imgright_data[row * d_cols + d_cols - 1]);
		}
		else
		{
			for(i = 0; i < pixels_number_thread; i++)
			{
				col = col_start + i;
				d_imgleft_grad[row * d_cols + col] = d_tab[ (d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1]) * 2 \
													    + d_imgleft_data[(row - pre_row_add) * d_cols + col + 1] - d_imgleft_data[(row - pre_row_add) * d_cols + col - 1] \
													    + d_imgleft_data[(row + next_row_add) * d_cols + col + 1] - d_imgleft_data[(row + next_row_add) * d_cols + col - 1] \
													   ];
				d_imgright_grad[row * d_cols + col] = d_tab[ (d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1]) * 2 \
													    + d_imgright_data[(row - pre_row_add) * d_cols + col + 1] - d_imgright_data[(row - pre_row_add) * d_cols + col - 1] \
													    + d_imgright_data[(row + next_row_add) * d_cols + col + 1] - d_imgright_data[(row + next_row_add) * d_cols + col - 1] \
													   ];
				//printf("row=%d, x=%d, right_data=%d\n", row, col, d_imgright_data[row * d_cols + col]);
			//	printf("row=%d,col=%d,grad=%d, pre_row=%d, org_imgdata=%d\n", row, col, d_imgleft_grad[row * d_cols + col], (row - pre_row_add), d_imgleft_data[row * d_cols + col ]);
				//printf("row=%d,col=%d,grad=%d, pre_row=%d, org_imgdata=%d\n", row, col, d_imgright_grad[row * d_cols + col], (row - pre_row_add), d_imgright_data[row * d_cols + col ]);
				//printf("y=%d,%d=%d\n", row, col,  d_imgright_grad[row * d_cols + col]);
			//	if(0 == row && col == 442)
			//	{
			//		printf("index 1=%d,2=%d,3=%d,4=%d,5=%d,6=%d, pre_now_add=%d, next_row_add=%d\n", d_imgright_data[row * d_cols + col + 1] , d_imgright_data[row * d_cols + col - 1], d_imgright_data[(row - pre_row_add) * d_cols + col + 1], d_imgright_data[(row - pre_row_add) * d_cols + col - 1], d_imgright_data[(row + next_row_add) * d_cols + col + 1], d_imgright_data[(row + next_row_add) * d_cols + col - 1], pre_row_add, next_row_add);
			//		printf("src_data 1=%d,2=%d,3=%d,4=%d,5=%d,6=%d, d_cols=%d\n", row * d_cols + col + 1, row * d_cols + col - 1, (row - pre_row_add) * d_cols + col + 1, (row - pre_row_add) * d_cols + col - 1, (row + next_row_add) * d_cols + col + 1, (row + next_row_add) * d_cols + col - 1,d_cols);
			//	}
			}
		}

	}

}
*/


__global__ void get_gradient(PixType *d_imgleft_data, PixType *d_imgright_data, PixType *d_imgleft_grad, PixType *d_imgright_grad, PixType *d_tab, int d_rows, int d_cols)
{
	int row = threadIdx.x;  //该线程所在行
	int col = 0;
	int pre_row_add = 0, next_row_add = 0;

	if(0 == row)//第一行
	{
		pre_row_add = 0;
		next_row_add = 1;
	}
	else if(d_rows -1 == row) //最后一行
	{
		pre_row_add = 1;
		next_row_add = 0;
	}
	else
	{
		pre_row_add = 1;
		next_row_add = 1;
	}

	for(col = 1; col < d_cols - 1; col++)
	{
		d_imgleft_grad[row * d_cols + col] = d_tab[ (d_imgleft_data[row * d_cols + col + 1] - d_imgleft_data[row * d_cols + col - 1]) * 2 \
											 + d_imgleft_data[(row - pre_row_add) * d_cols + col + 1] - d_imgleft_data[(row - pre_row_add) * d_cols + col - 1] \
											 + d_imgleft_data[(row + next_row_add) * d_cols + col + 1] - d_imgleft_data[(row + next_row_add) * d_cols + col - 1] \
		];
		d_imgright_grad[row * d_cols + col] = d_tab[ (d_imgright_data[row * d_cols + col + 1] - d_imgright_data[row * d_cols + col - 1]) * 2 \
											  + d_imgright_data[(row - pre_row_add) * d_cols + col + 1] - d_imgright_data[(row - pre_row_add) * d_cols + col - 1] \
											  + d_imgright_data[(row + next_row_add) * d_cols + col + 1] - d_imgright_data[(row + next_row_add) * d_cols + col - 1] \
		];
	}


	__syncthreads();

	d_imgleft_grad[row * d_cols + 0] = d_tab[0];
	d_imgright_grad[row * d_cols + 0] = d_tab[0];


	d_imgleft_data[row * d_cols + 0] = d_tab[0];
	d_imgright_data[row * d_cols + 0] = d_tab[0]; //此处仿照opencv代码，感觉opencv不对

	d_imgleft_grad[row * d_cols + d_cols - 1] = d_tab[0];
	d_imgright_grad[row * d_cols + d_cols - 1] = d_tab[0];

	d_imgleft_data[row * d_cols + d_cols - 1] = d_tab[0];
	d_imgright_data[row * d_cols + d_cols - 1] = d_tab[0]; //此处仿照opencv代码，感觉opencv不对

}


//每行有MAX_DISPARITY个线程，每个线程处理该行所有点的特定视差
__global__ void get_pixel_diff(const PixType * d_imgleft_buf, const PixType * d_imgright_buf, int rows, int cols, int diff_scale, CostType *d_cost) 
{
	int row = blockIdx.x;
	int now_disparity = threadIdx.x;

	const PixType *local_imgleft_buf = d_imgleft_buf + row * cols;
	const PixType *local_imgright_buf = d_imgright_buf + row * cols; //找到该行起始位置对应的地址
	
	for(int x = MAX_DISPARITY; x < cols; x++) //最大视差作为左图的起始点
	{
		int u = local_imgleft_buf[x];
		int ul = x > 0 ? (u + local_imgleft_buf[x - 1])/2 : u;
		int ur = x < cols - 1 ? (u + local_imgleft_buf[x + 1])/2 : u;	

		int u0 = min(ul, ur); u0 = min(u0, u);
		int u1 = max(ul, ur); u1 = max(u1, u); 


		int v = local_imgright_buf[x - now_disparity];
		int vl = x >= now_disparity + 1 ? (local_imgright_buf[x - now_disparity - 1] + v)/2 : v;
		int vr = x < cols + now_disparity - 1 ? (local_imgright_buf[x - now_disparity + 1] + v)/2 : v;

		int v0 = min(vl, vr); v0 = min(v0, v);
		int v1 = max(vl, vr); v1= max(v1, v);

		int c0 = max(0, u - v1); c0 = max(c0, v0 - u);
		int c1 = max(0, v - u1); c1 = max(c1, u0 - v);

		int pre_cost = d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity];
   		d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity] = d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity] + (min(c0, c1) >> diff_scale);
		//if(row ==479 && now_disparity == 0 && x == MAX_DISPARITY+1)
		//if(row ==479 && x == 128)
		//{
		//	printf("large row=%d, cols=%d, d=%d, d_pixdiff=%d, pixdiff_left=%d, pixdif_right(%d-%d)=%d ul=%d, ur=%d, vl=%d, vr=%d, v0=%d,v1=%d, u0=%d, u1=%d, pre_cost=%d, c0=%d, c1=%d, ur_sub=%d,diff_scale=%d\n", row, x, now_disparity, d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity], u, x, now_disparity,v, ul, ur, vl, vr, v0, v1, u0, u1, pre_cost,c0,c1, local_imgright_buf[x+1],diff_scale);

		//}
		/*
		if(diff_scale == 0)
		{
			printf("row=%d, cols=%d, d=%d, d_cost=%d, pixdiff_left=%d, pixdif_right(%d-%d)=%d ul=%d, ur=%d, vl=%d, vr=%d, lmin=%d,lmax=%d, u0=%d, u1=%d, pre_cost=%d, c0=%d, c1=%d\n", row, x, now_disparity, d_cost[(row * cols + x) * MAX_DISPARITY + now_disparity], u, x, MAX_DISPARITY,v, ul, ur, vl, vr, v0, v1, u0, u1, pre_cost,c0,c1);
		//	printf("y=%d,%d=%d, d=%d src=%d %d\n", row, x-now_disparity, v, now_disparity,d_imgright_buf[row * cols + x-now_disparity], local_imgright_buf[x-now_disparity] );
	    }
		*/
	}
}

__global__ void get_hsum(const CostType *d_pixel_diff, CostType *d_hsum, int rows, int cols, int blocksize)
{
	int row = blockIdx.x; //该线程所处理的行
	int now_disparity = threadIdx.x; //该线程处理该行的某个视差
	int max_col = cols - MAX_DISPARITY;  //对比sgbm中的width1
	int SH2 = blocksize/2, SW2=  blocksize/2;

	CostType *local_hsumAdd = d_hsum + (row * cols  + MAX_DISPARITY)* MAX_DISPARITY; // d_hsum的前MAX_DISPARITY列无法计算
	const CostType *local_pixel_diff = d_pixel_diff + (row * cols + MAX_DISPARITY) * MAX_DISPARITY; //d_pixel_diff的前MAX_DISPARITY列没有计算
	
	//可以计算的左边第一个像素
	local_hsumAdd[now_disparity] = 0;
	for(int x = 0; x <=SW2 * MAX_DISPARITY; x += MAX_DISPARITY) 
	{
		int scale = x == 0 ? SW2 + 1 : 1;
		local_hsumAdd[now_disparity] = local_hsumAdd[now_disparity] + local_pixel_diff[x + now_disparity] * scale;
	//	if(row == 479 && now_disparity == 0)
	//	printf("row=%d,d=%d,rang_x=%d, scale=%d, pixdiff=%d, src_pixdiff=%d,hsumadd=%d\n", row, now_disparity,x/MAX_DISPARITY, scale, local_pixel_diff[x + now_disparity], d_pixel_diff[row * cols *MAX_DISPARITY + MAX_DISPARITY * MAX_DISPARITY + now_disparity ], local_hsumAdd[now_disparity]);
	}
	//printf("final row=%d,d=%d,hsumAdd=%d\n", row, now_disparity, local_hsumAdd[now_disparity]);

	for(int x = MAX_DISPARITY; x < max_col  * MAX_DISPARITY; x += MAX_DISPARITY)
	{
		const CostType *pixAdd = local_pixel_diff + min(x + SW2 * MAX_DISPARITY, (max_col - 1) * MAX_DISPARITY);
		const CostType *pixSub = local_pixel_diff + max(x - (SW2 + 1) * MAX_DISPARITY, 0);
		local_hsumAdd[x + now_disparity] = local_hsumAdd[x - MAX_DISPARITY + now_disparity] + pixAdd[now_disparity] - pixSub[now_disparity];

		//if(row == 479 && now_disparity == 0)
		//	printf("row=%d,x=%d,d=%d,hsumAdd=%d, pixSub=%d, pixAdd=%d, hsumAdd_pre=%d, pixAddIndex=%d, SW2=%d, max_col=%d, x=%d\n", row, x/MAX_DISPARITY + MAX_DISPARITY, now_disparity, local_hsumAdd[x+now_disparity], pixSub[now_disparity],pixAdd[now_disparity], local_hsumAdd[x - MAX_DISPARITY + now_disparity], min(x + SW2 * MAX_DISPARITY, (max_col - 1) * MAX_DISPARITY), SW2, max_col, x);
	}
}

__global__ void get_cost(const CostType *d_hsumAdd, CostType *d_cost, int p2, int rows, int cols, int blocksize)
{
	int col = blockIdx.x + MAX_DISPARITY; //线程块代表每一列, 因为只开了cols - MAX_DISPARITY个线程块
	int now_disparity = threadIdx.x; //线程块中的一个线程处理某一列的视差为d时的代价
	
	CostType *local_cost = d_cost + col * MAX_DISPARITY; //第一行该列的地址
	const CostType *local_hsumAdd = d_hsumAdd + col * MAX_DISPARITY; //第一行该列的地址
	int SH2 = blocksize/2, SW2=  blocksize/2;
	
	//y == 0
	local_cost[0 + now_disparity] = p2;
	for(int i = 0; i <= SH2; i++)
	{
		int scale = i == 0 ? SH2 + 1 : 1;
		local_cost[now_disparity] = local_cost[0 + now_disparity] + local_hsumAdd[i * cols * MAX_DISPARITY + now_disparity] * scale;
		//if(col == 128 && now_disparity == 127)
		//	printf("k=%d,col=%d,d=%d,hsumAdd=%d,scale=%d,C=%d\n", i,col,now_disparity,local_hsumAdd[i * cols * MAX_DISPARITY + now_disparity],scale,local_cost[now_disparity]);
	}
 
	if(128 == col)  //copy opencv, 最左边第一列和第一行保持一致
	{
		int k = 0;
		for(k = 1 + SH2; k < rows; k++)
		{
			int y = k - SH2;
			local_cost[cols * MAX_DISPARITY * y + now_disparity] = local_cost[cols * MAX_DISPARITY * (y - 1) + now_disparity];
		}

		for(int y = k - SH2; y < rows; y++)
		{
			local_cost[cols * MAX_DISPARITY * y + now_disparity] = local_cost[cols * MAX_DISPARITY * (y - 1) + now_disparity];
		}
	}
	else
	{
		int k = 0;
		for(k = 1 + SH2; k < rows; k++)
		{
			int y = k - SH2;
			const CostType *h_sumSub = local_hsumAdd + (k >= blocksize ? cols * MAX_DISPARITY * (k - blocksize) : 0);

			local_cost[cols * MAX_DISPARITY * y + now_disparity] = local_cost[cols * MAX_DISPARITY * (y - 1) + now_disparity] + local_hsumAdd[cols * MAX_DISPARITY * k + now_disparity] - h_sumSub[now_disparity];
		//	if(475 == y && col == 129)
		//	{
		//		printf("y=%d,col=%d,d=%d,c_pre=%d,hsumadd=%d,hsumsub=%d,c=%d, src_hsumadd=%d, src_hsumsub=%d\n", y, col, now_disparity,  local_cost[cols * MAX_DISPARITY * (y - 1) + now_disparity], local_hsumAdd[cols * MAX_DISPARITY * k + now_disparity],  h_sumSub[now_disparity], local_cost[cols * MAX_DISPARITY * y + now_disparity] , d_hsumAdd[k*cols * MAX_DISPARITY + col * MAX_DISPARITY + now_disparity], d_hsumAdd[col * MAX_DISPARITY]);
		//	}
		}
		
		for(int y = k - SH2; y < rows; y++)  //fill the last rows with previous value
		{
			local_cost[cols * MAX_DISPARITY *y + now_disparity] = local_cost[cols * MAX_DISPARITY * (y - 1) + now_disparity];	
		}
	}
}

__global__ void fill_tab(PixType *d_tab, int TAB_SIZE, int TAB_OFS, int ftzero)
{
	for(int k = 0; k < TAB_SIZE; k++)
	{
		d_tab[k] = (PixType)(min(max(k - TAB_OFS, -ftzero), ftzero) + ftzero); 
	}
}
