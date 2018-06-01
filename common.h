#ifndef _COMMON
#define _COMMON

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <assert.h>
#include <algorithm>

#define WARP_SIZE 32

#define MAX_DISPARITY 96
#define INVALID_DISP_SCALED -16
#define DISP_SCALE 16
#define DISP_SHIFT 4

using namespace std;

typedef unsigned char PixType;
typedef short CostType;
typedef short DispType;

typedef struct _SGM_PARAMS
{
	int P1;
	int P2;
	int preFilterCap;
	int BlockSize;
	int uniquenessRatio;
	int disp12MaxDiff;
}SGM_PARAMS;


#endif
