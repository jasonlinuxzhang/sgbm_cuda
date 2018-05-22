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
extern int sgbm( cv::Mat image_left, cv::Mat image_right, cv::Mat &disparity_im);
void zy_remap(cv::Mat &img1, cv::Mat &img2);
static cv::Point pt;
static bool mouseFlag = false;
static cv::Mat xyz;

void printHelp();
void outputImageName(int num);

int main(int argc, char **argv) {

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
	params.P1 = 2;
	params.P2 = 2;
	cuda_init(&params);

	cv::Mat resizeImg_left, resizeImg_right;

    while (key != 27) {

		if('l' == key )		
			time_print = !(time_print);

		if(time_print)
		{
			frame_start = cv::getTickCount();
		}

        sprintf(left_img_name, "%s/left%d.jpg", prefix, i);
        sprintf(right_img_name, "%s/right%d.jpg", prefix, i);
        resizeImg_left = cv::imread(left_img_name, -1);
        if(resizeImg_left.empty())
        {
            std::cout<<"read "<<left_img_name<<" fail\n";
            return 1;
        }
        resizeImg_right = cv::imread(right_img_name, -1);
        if(resizeImg_right.empty())
        {
            std::cout<<"read "<<right_img_name<<" fail\n";
            return 1;
        }
		i++;

		cv::imshow("left_zed", resizeImg_left);
		cv::imshow("right_zed", resizeImg_right);
		
		zy_remap(resizeImg_left, resizeImg_right);

		if(time_print)
		{
			frame_end = cv::getTickCount();
			std::cout<<"One Process Need:"<<(frame_end - frame_start)*1000/cv::getTickFrequency()<<" ms"<<std::endl;
		}

		key = cv::waitKey(10000);
    }

	free_gpu_mem();
    return 0;
}


static void on_mouse( int event, int x, int y, int flags, void* ustc)
{  
    if((event == CV_EVENT_LBUTTONDOWN))
    {  
		mouseFlag = true;
        pt = cv::Point(x,y);
    }   
}

void zy_remap(cv::Mat &img1, cv::Mat &img2)
{
	string intrinsic_filename = "intrinsics.yml", extrinsic_filename = "extrinsics.yml";	

    cv::Rect roi1, roi2;
    cv::Mat Q;
    cv::Size img_size = img1.size();
	float scale = 1.0;
	cv::Mat disp;

    if( !intrinsic_filename.empty() )
    {
        // reading intrinsic parameters
        cv::FileStorage fs(intrinsic_filename, cv::FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            exit(0);
        }

        cv::Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, cv::FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            exit(0);
        }

        cv::Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        cv::Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        cv::Mat img1r, img2r;
        remap(img1, img1r, map11, map12, cv::INTER_LINEAR);
        remap(img2, img2r, map21, map22, cv::INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }
	
	compute_disparity(&img1, &img2, NULL);

}
