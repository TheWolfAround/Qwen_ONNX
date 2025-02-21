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

# Third Party Includes
include_directories(${THIRD_PARTY_FOLDER}/)

if (MSVC)
	# Third Party Includes
	include_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/include)

	link_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/lib)

	set(ONNX_RUNTIME_LIBS onnxruntime)

    # set(ONNX_RUNTIME_LIBS
	# 	onnxruntime,
	# 	onnxruntime_providers_cuda,
	# 	onnxruntime_providers_tensorrt)
endif ()

if (UNIX)

endif ()

# GLM
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

add_executable(
	${PROJECT_NAME}

	${WORKSPACE_DIR}/source/main.cpp)

target_link_libraries(
	${PROJECT_NAME}

	PRIVATE

	${ONNX_RUNTIME_LIBS})

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
    ${ONNX_LIBS_FOLDER}/onnxruntime_providers_cuda.dll
    ${ONNX_LIBS_FOLDER}/onnxruntime_providers_tensorrt.dll
    ${ONNX_LIBS_FOLDER}/onnxruntime_providers_shared.dll)

set(DESTINATION_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Call the function with the list of source files and destination directory
copy_files_if_not_exist("${ONNX_DLL_FILES}" "${DESTINATION_DIR}")

# end of file
