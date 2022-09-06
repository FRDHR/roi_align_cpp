cmake_minimum_required (VERSION 3.22.0)
project(roi_align  LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations  " )
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -mavx -mavx2 -mfma -fPIC -Werror=return-type -Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
set(CNPY_HOME "/opt/cnpy/")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CNPY_HOME}/include)

set(COMMON_LIBS ${glog_LIBRARIES}
    gflags
    ${OpenCV_LIBS}
    ${CNPY_HOME}/lib/libcnpy.so
)

link_directories(${CNPY_HOME}/lib)

file(GLOB MAIN_SRCS "${CMAKE_SOURCE_DIR}/src/*.cc")
add_executable(${CMAKE_PROJECT_NAME} ${MAIN_SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${COMMON_LIBS})
