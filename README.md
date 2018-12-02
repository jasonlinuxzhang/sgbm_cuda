# zhangyou_sgm_cuda

Opencv does not implement the gpu version of the sgm algorithm, this project is based on the opencv sgm algorithm. After my test, it is 3-4 times faster than the cpu version of the opencv sgm algorithm, and the effect is exactly the same as opencv sgm.



How to compile and use

mkdir build
cd build
cmake ..
make
Regarding the use of the program, you can look at the program logic in the main function, which is very simple.


Finally
The program is based on opencv3.4 and cuda9.0, and the program assumes that the left and right images have been corrected.

If you find this program useful, please give me a star, thank you.
