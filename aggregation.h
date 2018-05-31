#ifndef _AGGREGATION
#define _AGGREGATION
#include "common.h"
__global__ void cost_aggregation_lr(const CostType *d_cost, CostType *sp, int p1, int p2, int cols, int rows);
__global__ void cost_aggregation_ud_lr(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows);
__global__ void cost_aggregation_ud(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows);
__global__ void cost_aggregation_rl_ud(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows);

__global__ void cost_aggregation_rl(const CostType *d_cost, CostType *d_sp, int p1, int p2, int cols, int rows);
__global__ void get_disparity(const CostType *d_sp, DispType *d_disp, CostType *d_mins, int uniquenessRatio, CostType *disp2cost, DispType *disp2, int cols, int rows);

__global__ void lrcheck(DispType * d_disp, const CostType * d_mins, DispType *disp2, CostType *disp2cost, int disp12MaxDiff, int cols, int rows);

__global__ void MedianFilter(const DispType *d_input, DispType *d_out, int rows, int cols);
#endif
