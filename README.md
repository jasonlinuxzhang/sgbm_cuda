# zhangyou_sgm_cuda

Opencv does not implement the gpu version of the sgm algorithm, this project is based on the opencv sgm algorithm. After my test, it is 3-4 times faster than the cpu version of the opencv sgm algorithm, and the effect is exactly the same as opencv sgm. The maximum disparity can be set to an integer multiple of 32, the maximum is 256.



**How to compile and use** 

mkdir build 

cd build 

cmake .. 

make 

Regarding the use of the program, you can look at the program logic in the main function, which is very simple. 


**Finally**

The program is based on opencv3.4 and cuda9.0, and the program assumes that the left and right images have been corrected.

If you find this program useful, please give me a star, thank you.


**Program performance display**

If you want to watch the recorded demo video, click
[Road effect][1]


or
![example1]https://github.com/jasonlinuxzhang/sgm_cuda/blob/master/example1.jpg
![example2]https://github.com/jasonlinuxzhang/sgm_cuda/blob/master/example2.jpg


  [1]: https://v.youku.com/v_show/id_XMzk0NjE1MDEwNA==.html?spm=a2hzp.8253869.0.0
