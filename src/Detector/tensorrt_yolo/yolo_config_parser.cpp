/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "yolo_config_parser.h"

#include <assert.h>
#include <iostream>

//DEFINE_string(network_type, "not-specified",
//              "[REQUIRED] Type of network architecture. Choose from yolov2, yolov2-tiny, "
//              "yolov3 and yolov3-tiny");
//DEFINE_string(config_file_path, "not-specified", "[REQUIRED] Darknet cfg file");
//DEFINE_string(wts_file_path, "not-specified", "[REQUIRED] Darknet weights file");
//DEFINE_string(labels_file_path, "not-specified", "[REQUIRED] Object class labels file");
//DEFINE_string(precision, "kFLOAT",
//              "[OPTIONAL] Inference precision. Choose from kFLOAT, kHALF and kINT8.");
//DEFINE_string(deviceType, "kGPU",
//              "[OPTIONAL] The device that this layer/network will execute on. Choose from kGPU and kDLA(only for kHALF).");
//DEFINE_string(calibration_table_path, "not-specified",
//              "[OPTIONAL] Path to pre-generated calibration table. If flag is not set, a new calib "
//              "table <network-type>-<precision>-calibration.table will be generated");
//DEFINE_string(engine_file_path, "not-specified",
//              "[OPTIONAL] Path to pre-generated engine(PLAN) file. If flag is not set, a new "
//              "engine <network-type>-<precision>-<batch-size>.engine will be generated");
//DEFINE_string(input_blob_name, "data",
//              "[OPTIONAL] Name of the input layer in the tensorRT engine file");
//DEFINE_bool(print_perf_info, false, "[OPTIONAl] Print performance info on the console");
//DEFINE_bool(print_prediction_info, false, "[OPTIONAL] Print detection info on the console");
//DEFINE_string(
//    test_images, "data/test_images.txt",
//    "[REQUIRED] Text file containing absolute paths or filenames of all the images to be "
//    "used for inference. If only filenames are provided, their corresponding source directory "
//    "has to be provided through 'test_images_path' flag");
//DEFINE_string(test_images_path, "not-specified",
//              "[OPTIONAL] absolute source directory path of the list of images supplied through "
//              "'test_images' flag");
//DEFINE_string(calibration_images, "data/calibration_images.txt",
//              "[OPTIONAL] Text file containing absolute paths or filenames of calibration images. "
//              "Flag required if precision is kINT8 and there is not pre-generated calibration "
//              "table. If only filenames are provided, their corresponding source directory has to "
//              "be provided through 'calibration_images_path' flag");
//DEFINE_string(calibration_images_path, "not-specified",
//              "[OPTIONAL] absolute source directory path of the list of images supplied through "
//              "'calibration_images' flag");
//DEFINE_uint64(batch_size, 1, "[OPTIONAL] Batch size for the inference engine.");
//DEFINE_double(prob_thresh, 0.01, "[OPTIONAL] Probability threshold for detected objects");
//DEFINE_double(nms_thresh, 0.5, "[OPTIONAL] IOU threshold for bounding box candidates");
//DEFINE_bool(do_benchmark, false,
//            "[OPTIONAL] Generate JSON file with detection info in coco benchmark format");
//DEFINE_bool(save_detections, false,
//            "[OPTIONAL] Flag to save images overlayed with objects detected.");
//DEFINE_bool(view_detections, false,
//            "[OPTIONAL] Flag to view images overlayed with objects detected.");
//DEFINE_string(save_detections_path, "not-specified",
//              "[OPTIONAL] Path where the images overlayed with bounding boxes are to be saved");
//DEFINE_bool(
//    decode, true,
//    "[OPTIONAL] Decode the detections. This can be set to false if benchmarking network for "
//    "throughput only");
//DEFINE_uint64(seed, std::time(0), "[OPTIONAL] Seed for the random number generator");
//DEFINE_bool(shuffle_test_set, false,
//            "[OPTIONAL] Shuffle the test set images before running inference");
//
static bool isFlagDefault(std::string flag) { return flag == "not-specified" ? true : false; }

static bool networkTypeValidator(const char* flagName, std::string value)
{
	/*if (((FLAGS_network_type) == "yolov2") || ((FLAGS_network_type) == "yolov2-tiny")
		|| ((FLAGS_network_type) == "yolov3") || ((FLAGS_network_type) == "yolov3-tiny"))
		return true;

	else
		std::cout << "Invalid value for --" << flagName << ": " << value << std::endl;
*/
    return false;
}

static bool precisionTypeValidator(const char* flagName, std::string value)
{
  /*  if ((FLAGS_precision == "kFLOAT") || (FLAGS_precision == "kINT8")
        || (FLAGS_precision == "kHALF"))
        return true;
    else
        std::cout << "Invalid value for --" << flagName << ": " << value << std::endl;*/
    return false;
}

static bool verifyRequiredFlags()
{
	/* assert(!isFlagDefault(FLAGS_network_type)
			&& "Type of network is required and is not specified.");
	 assert(!isFlagDefault(FLAGS_config_file_path)
			&& "Darknet cfg file path is required and not specified.");
	 assert(!isFlagDefault(FLAGS_wts_file_path)
			&& "Darknet weights file is required and not specified.");
	 assert(!isFlagDefault(FLAGS_labels_file_path) && "Lables file is required and not specified.");
	 assert((FLAGS_wts_file_path.find(".weights") != std::string::npos)
			&& "wts file not recognised. File needs to be of '.weights' format");
	 assert((FLAGS_config_file_path.find(".cfg") != std::string::npos)
			&& "config file not recognised. File needs to be of '.cfg' format");
	 if (!(networkTypeValidator("network_type", FLAGS_network_type)
		   && precisionTypeValidator("precision", FLAGS_precision)))
		 return false;
 */
    return true;
}

//void yoloConfigParserInit(int argc, char** argv)
//{
	/*gflags::ParseCommandLineFlags(&argc, &argv, false);
	assert(verifyRequiredFlags());

	FLAGS_calibration_images_path
		= isFlagDefault(FLAGS_calibration_images_path) ? "" : FLAGS_calibration_images_path;
	FLAGS_test_images_path = isFlagDefault(FLAGS_test_images_path) ? "" : FLAGS_test_images_path;

	if (isFlagDefault(FLAGS_engine_file_path))
	{
		int npos = FLAGS_wts_file_path.find(".weights");
		assert(npos != std::string::npos
			   && "wts file file not recognised. File needs to be of '.weights' format");
		std::string dataPath = FLAGS_wts_file_path.substr(0, npos);
		FLAGS_engine_file_path = dataPath + "-" + FLAGS_precision + "-" + FLAGS_deviceType + "-batch"
			+ std::to_string(FLAGS_batch_size) + ".engine";
	}

	if (isFlagDefault(FLAGS_calibration_table_path))
	{
		int npos = FLAGS_wts_file_path.find(".weights");

		assert(npos != std::string::npos
			   && "wts file file not recognised. File needs to be of '.weights' format");
		std::string dataPath = FLAGS_wts_file_path.substr(0, npos);
		FLAGS_calibration_table_path = dataPath + "-calibration.table";
	}*/
//}

//NetworkInfo getYoloNetworkInfo()
//{
	/*return NetworkInfo{FLAGS_network_type,     FLAGS_config_file_path, FLAGS_wts_file_path,
					   FLAGS_labels_file_path, FLAGS_precision,        FLAGS_deviceType,
					   FLAGS_calibration_table_path, FLAGS_engine_file_path, FLAGS_input_blob_name};*/
//}

//InferParams getYoloInferParams()
//{

    //return InferParams{FLAGS_print_perf_info,    FLAGS_print_prediction_info,
    //                   FLAGS_calibration_images, FLAGS_calibration_images_path,
    //                   (float)FLAGS_prob_thresh,        (float)FLAGS_nms_thresh};
//}

//uint64_t getSeed() { return FLAGS_seed; }
//
//std::string getNetworkType() { return FLAGS_network_type; }
//
//std::string getPrecision() { return FLAGS_precision; }
//
//std::string getTestImages()
//{
//    size_t extIndex = FLAGS_test_images.find_last_of(".txt");
//    assert(extIndex != std::string::npos
//           && "test_images file not recognised. File needs to be of type '.txt' format");
//    return FLAGS_test_images;
//}
//
//std::string getTestImagesPath() { return FLAGS_test_images_path; }
//
//bool getDecode() { return FLAGS_decode; }
//bool getDoBenchmark() { return FLAGS_do_benchmark; }
//bool getViewDetections() { return FLAGS_view_detections; }
//bool getSaveDetections()
//{
//    if (FLAGS_save_detections)
//        assert(!isFlagDefault(FLAGS_save_detections_path)
//               && "save_detections path has to be set if save_detections is set to true");
//    return FLAGS_save_detections;
//}
//
//std::string getSaveDetectionsPath() { return FLAGS_save_detections_path; }
//
//uint32_t getBatchSize() { return FLAGS_batch_size; }
//
//bool getShuffleTestSet() { return FLAGS_shuffle_test_set; }
