cd ..
md build
cd build
cmake.exe . ..  -G "Visual Studio 16 2019" -A "x64" ^
          -DOpenCV_DIR=C:/work/libraries/opencv/opencv_64_cuda ^
          -DUSE_OCV_BGFG=ON ^
          -DUSE_OCV_KCF=ON  ^
          -DUSE_OCV_UKF=ON  ^
          -DSILENT_WORK=OFF ^
          -DBUILD_EXAMPLES=ON ^
          -DBUILD_ASYNC_DETECTOR=ON ^
          -DBUILD_CARS_COUNTING=ON ^
          -DBUILD_YOLO_LIB=ON ^
          -DCUDNN_INCLUDE_DIR=C:/cudnn-10.0-windows10-x64-v7.6.5.32/cuda/include ^
          -DCUDNN_LIBRARY=C:/cudnn-10.0-windows10-x64-v7.6.5.32/cuda/lib/x64/cudnn.lib ^
          -DBUILD_YOLO_TENSORRT=ON ^
          -DTensorRT_LIBRARY=C:/TensorRT-5.1.5.0/lib/*.lib ^
          -DTensorRT_INCLUDE_DIR=C:/TensorRT-5.1.5.0/include
cmake.exe --build . -j 6 --config Release
