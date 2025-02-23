cmake_minimum_required(VERSION 3.25)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

project(Qwen_ONNX)

set(CMAKE_CXX_STANDARD 17)

if (EDITOR_BUILD)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
	if (MSVC)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP /O2 /MD /arch:AVX2")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /O2 /MD /arch:AVX2")
	endif ()

	if (UNIX)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -march=native")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")
	endif ()
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	if (MSVC)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP /Od /MDd")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /Od /MDd")
	endif ()

	if (UNIX)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -Wall -ggdb")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -ggdb")
	endif ()
endif ()

set(WORKSPACE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_FOLDER ${WORKSPACE_DIR}/__third_parties__)

# Project Includes
include_directories(${WORKSPACE_DIR}/include)
include_directories(${WORKSPACE_DIR}/include/onnx)

# Third Party Includes

if (MSVC)
	include_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/include)
	include_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/include/core)
	include_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/include/core/providers)
	include_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/include/core/session)

	link_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/lib)

    set(ONNX_RUNTIME_LIB onnxruntime)
endif ()

if (UNIX)

endif ()

add_executable(
	${PROJECT_NAME}

	${WORKSPACE_DIR}/source/main.cpp
	${WORKSPACE_DIR}/source/onnx/onnxruntime.cpp)

target_link_libraries(
	${PROJECT_NAME}

	PRIVATE

	${ONNX_RUNTIME_LIB})

# Define a function to copy files if they don't exist in the destination
function(copy_files_if_not_exist SRC_FILES DST_DIR)
    # Ensure the destination directory exists
    file(MAKE_DIRECTORY ${DST_DIR})

    # Iterate over each file in the source list
    foreach(SRC_FILE IN LISTS SRC_FILES)
        # Get the filename (basename) of the source file
        get_filename_component(FILENAME ${SRC_FILE} NAME)

        # Construct the full path for the destination file
        set(DST_FILE "${DST_DIR}/${FILENAME}")

        # Check if the file does NOT exist in the destination
        if(NOT EXISTS ${DST_FILE})
            # If it doesn't exist, copy the file from source to destination
            message(STATUS "Copying ${SRC_FILE} to ${DST_FILE}")
            configure_file(${SRC_FILE} ${DST_FILE} COPYONLY)
        else()
            # If it exists, print a message indicating that the file was skipped
            message(STATUS "File ${DST_FILE} already exists. Skipping.")
        endif()
    endforeach()
endfunction()

set(ONNX_LIBS_FOLDER ${THIRD_PARTY_FOLDER}/onnxruntime/windows/lib)

# Example usage:
# Assuming you have a list of files and a destination directory
set(ONNX_DLL_FILES
    ${ONNX_LIBS_FOLDER}/onnxruntime.dll
    ${ONNX_LIBS_FOLDER}/DirectML.dll
    ${ONNX_LIBS_FOLDER}/DirectML.Debug.dll
    ${ONNX_LIBS_FOLDER}/onnxruntime_providers_cuda.dll
    ${ONNX_LIBS_FOLDER}/onnxruntime_providers_tensorrt.dll
    ${ONNX_LIBS_FOLDER}/onnxruntime_providers_shared.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cublas64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cublasLt64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cudart64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cudnn64_8.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cufft64_11.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cufftw64_11.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cuinj64_126.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/curand64_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cusolver64_11.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cusolverMg64_11.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/cusparse64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppc64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppial64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppicc64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppidei64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppif64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppig64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppim64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppist64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppisu64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nppitc64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/npps64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvblas64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvfatbin_120_0.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvinfer_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvinfer_builder_resource_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvinfer_dispatch_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvinfer_lean_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvinfer_plugin_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvinfer_vc_plugin_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvJitLink_120_0.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvjpeg64_12.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvonnxparser_10.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvrtc-builtins64_126.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvrtc64_120_0.alt.dll
	${ONNX_LIBS_FOLDER}/cuda_tensorrt/nvrtc64_120_0.dll)

set(DESTINATION_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Call the function with the list of source files and destination directory
copy_files_if_not_exist("${ONNX_DLL_FILES}" "${DESTINATION_DIR}")

# end of file
