#include <iostream>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

// Sample includes
#include <time.h>

#include "common.h"

using namespace std;

void free_gpu_mem();
cv::Mat compute_disparity(cv::Mat *left_img, cv::Mat *right_img, float *cost_time);
void cuda_init(SGM_PARAMS *params);
void zy_remap(cv::Mat &img1, cv::Mat &img2);


string intrinsic_filename = "intrinsics.yml", extrinsic_filename = "extrinsics.yml";

cv::Rect roi1, roi2;
cv::Mat Q;
cv::Size img_size;
float scale = 1.0;
cv::Mat map11, map12, map21, map22;


int main(int argc, char **argv) {

	if(argc != 4)
	{
		std::cout<<"argc wrong\nuseage: ./a.out image_dir start_number end_number";
		return -1;
	}
    char *prefix = argv[1];
    int start_num = atoi(argv[2]);
    int end_num = atoi(argv[3]);
    char left_img_name[128] = {0};
    char right_img_name[128] = {0};

    char key = ' ';
	double frame_start = 0, frame_end = 0;
	bool time_print = false;
	int i = start_num;	

	SGM_PARAMS params;
	params.preFilterCap = 63;
	params.BlockSize = 9;
	params.P1 = 8 * params.BlockSize * params.BlockSize;
	params.P2 = 32 * params.BlockSize * params.BlockSize;
	params.uniquenessRatio = 1;
	params.disp12MaxDiff = 15;
	cuda_init(&params);

	cv::Mat resizeImg_left, resizeImg_right;

    while (key != 27) {

		double start = cv::getTickCount();
        sprintf(left_img_name, "%s/left%d.jpg", prefix, i);
        sprintf(right_img_name, "%s/right%d.jpg", prefix, i);
        resizeImg_left = cv::imread(left_img_name, cv::IMREAD_GRAYSCALE);
        if(resizeImg_left.empty())
        {
            std::cout<<"read "<<left_img_name<<" fail\n";
            return 1;
        }
        resizeImg_right = cv::imread(right_img_name, cv::IMREAD_GRAYSCALE);
        if(resizeImg_right.empty())
        {
            std::cout<<"read "<<right_img_name<<" fail\n";
            return 1;
        }
		i++;

//		cv::imshow("left_zed", resizeImg_left);
//		cv::imshow("right_zed", resizeImg_right);

		double end = cv::getTickCount();
		printf("Pre Cost:%lf ms.\n", (end - start)* 1000/cv::getTickFrequency());
		
		zy_remap(resizeImg_left, resizeImg_right);

		end = cv::getTickCount();

		printf("Total Cost:%lf ms.\n", (end - start)* 1000/cv::getTickFrequency());

		key = cv::waitKey(1);
    }

	//free_gpu_mem();
    return 0;
}


void zy_remap(cv::Mat &img1, cv::Mat &img2)
{
	compute_disparity(&img1, &img2, NULL);
}
