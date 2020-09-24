# zhangyou_sgbm_cuda  

Opencv does not implement the gpu version of the sgbm algorithm, this project is based on the opencv sgbm algorithm. After my test, it is 3-4 times faster than the cpu version of the opencv sgbm algorithm, and the effect is exactly the same as opencv sgbm. The maximum disparity can be set to an integer multiple of 32, the maximum is 256.   

After other people’s tests, the input image size is recommended to be 640*480. I don't limit the size of the image in the program, but I don't know why other sizes can't work.

> NOTE: This bug was fixed partly in commit `df82a9f2c4c2369bd6b327def4da6ad928375c3e`, max image size = [2048, 1024]


**How to compile and use**  
mkdir build  
cd build    
cmake ..  
make   
Regarding the use of the program, you can look at the program logic in the main function, which is very simple.   


**Finally**  
The program is based on opencv3.4 and cuda9.0, and the program assumes that the left and right images have been corrected.   


**Program performance display**  
If you want to watch the recorded demo video, click
[Road effect][1] .This video is not subject to any post processing, such as acceleration  


or  
![example1](https://github.com/jasonlinuxzhang/sgm_cuda/blob/master/example1.jpg)  
![example2](https://github.com/jasonlinuxzhang/sgm_cuda/blob/master/example2.jpg)   


If you find this program useful, please give me a star, thank you.  


  [1]: https://v.youku.com/v_show/id_XMzk0NjE1MDEwNA==.html?spm=a2hzp.8253869.0.0
