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

	include_directories(${THIRD_PARTY_FOLDER}/tokenizers/include)

	link_directories(${THIRD_PARTY_FOLDER}/onnxruntime/windows/lib)
	link_directories(${THIRD_PARTY_FOLDER}/tokenizers/windows/lib/${CMAKE_BUILD_TYPE})
	
    set(ONNX_RUNTIME_LIB onnxruntime)
	set(TOKENIZERS_CPP_LIB
		tokenizers_cpp
		Ntdll
		Bcrypt)
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

	${ONNX_RUNTIME_LIB}
	${TOKENIZERS_CPP_LIB})

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
    ${ONNX_LIBS_FOLDER}/DirectML.dll)

set(DESTINATION_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Call the function with the list of source files and destination directory
copy_files_if_not_exist("${ONNX_DLL_FILES}" "${DESTINATION_DIR}")

# end of file
