#ifndef _COSTS
#define _COSTS

#include "common.h"

__global__ void get_gradient(PixType *d_imgleft_data, PixType *d_imgright_data, PixType *d_imgleft_grad, PixType *d_imgright_grad, PixType *d_tab, int d_rows, int d_cols);
__global__ void fill_tab(PixType *d_tab, int TAB_SIZE, int TAB_OFS, int ftzero);
__global__ void get_pixel_diff(const PixType * d_imgleft_buf, const PixType * d_imgright_buf, int rows, int cols, int diff_scale, CostType *d_cost); 
__global__ void get_hsum(const CostType *d_pixel_diff, CostType *d_hsumAdd, int rows, int cols, int blocksize);
__global__ void get_cost(const CostType *d_hsumAdd, CostType *d_cost, int p2, int rows, int cols, int blocksize);

#endif
