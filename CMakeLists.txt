SET(execName ZED_with_OpenCV)
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(${execName})

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 OLD)
	cmake_policy(SET CMP0015 OLD)
endif(COMMAND cmake_policy)

SET(EXECUTABLE_OUTPUT_PATH ".")

set(OpenCV_DIR "/opt/disk/install/share/OpenCV")

find_package(ZED 2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)

message(STATUS "CUDA:" ${CUDA_INCLUDE_DIRS})
message(STATUS "OpenCVInclude:" ${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCVInclude:" ${OPENCV_INCLUDE_DIRS})
message(STATUS "OpenCVLibraries:" ${OpenCV_LIBRARIES})
message(STATUS "OpenCVLibs:" ${OpenCV_LIBS})
message(STATUS "OpenCVLibDIR:" ${OpenCV_LIBRARY_DIRS})
set(OpenCV_LIBRARY_DIRS "/opt/disk/install/lib")


IF(NOT WIN32)
    add_definitions(-Wno-format-extra-args)
    SET(SPECIAL_OS_LIBS "pthread" "X11")
ENDIF()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})

FILE(GLOB SRC_FILES *.cpp)
FILE(GLOB CUDA_FILES *.cu)
FILE(GLOB HDR_FILES *.h*)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3
	#-gencode arch=compute_20,code=[sm_20,sm_21] 666 CHECK THIS DEPEND ON THE PLATFORM
	-gencode arch=compute_30,code=sm_30
	-gencode arch=compute_35,code=sm_35
	-gencode arch=compute_50,code=[sm_50,compute_50]
	-gencode arch=compute_52,code=[sm_52,compute_52]
	-gencode arch=compute_61,code=sm_61
	-gencode arch=compute_62,code=sm_62
)


add_definitions(-std=c++11 -g)
cuda_add_executable(${execName} ${CUDA_FILES} ${HDR_FILES} ${SRC_FILES})

TARGET_LINK_LIBRARIES(${execName}
                        ${OpenCV_LIBS}
                        ${ZED_LIBRARIES}
                        ${SPECIAL_OS_LIBS}
                        ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NPP_LIBRARIES_ZED} )
