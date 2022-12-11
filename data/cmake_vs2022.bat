cd ..
md build
cd build
cmake.exe . ..  -G "Visual Studio 17 2022" -A "x64" ^
          -DOpenCV_DIR=C:/work/libraries/opencv/opencv_64_2022 ^
          -DUSE_OCV_BGFG=ON ^
          -DUSE_OCV_KCF=ON  ^
          -DUSE_OCV_UKF=ON  ^
          -DSILENT_WORK=OFF ^
          -DBUILD_EXAMPLES=ON ^
          -DBUILD_ASYNC_DETECTOR=ON ^
          -DBUILD_CARS_COUNTING=ON ^
          -DBUILD_YOLO_LIB=ON ^
          -DCUDNN_INCLUDE_DIR=C:/cuda/cudnn-windows-x86_64-8.6.0.163_cuda11/include ^
          -DCUDNN_LIBRARY=C:/cuda/cudnn-windows-x86_64-8.6.0.163_cuda11/lib/x64/cudnn.lib ^
          -DBUILD_YOLO_TENSORRT=ON ^
          -DTensorRT_LIBRARY=C:/cuda/TensorRT-8.4.3.1/lib/*.lib ^
          -DTensorRT_INCLUDE_DIR=C:/cuda/TensorRT-8.4.3.1/include
cmake.exe --build . -j 6 --config Release
