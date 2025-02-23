///
/// @file main.cpp
/// @author TheWolfAround
/// @brief 
/// @version 1.0.0
/// @date 21/02/2025
/// @copyright (c) 2025 All rights reserved.
///

#include <iostream>
#include <memory>
#include <filesystem>

#include <onnxruntime_cxx_api.h>

#include "onnxruntime.hpp"

static const std::string CWD = std::filesystem::current_path().string();

int main()
{
    const std::string model_path = CWD + R"(\..\..\watanabe_onnx_float16\wtb_float16_2.onnx)";

    std::unique_ptr<TWA::onnxruntime> onnx{new TWA::onnxruntime(model_path)};
    onnx->set_onnx_threads(2);

    onnx->print_exucution_providers();
    onnx->detect_input_output_node_info();

    std::vector<int64_t> ids{151644, 8948, 198, 151645, 198, 151644, 872, 198, 2679, 358};
    std::vector<int64_t> atn(ids.size(), 1);
    std::vector<float> rsp{};

    size_t rsp_size = ids.size() * 151936;

    rsp.resize(rsp_size);

    int a = 0;

    for (size_t i = 0; i < rsp_size; i++)
    {
        if (rsp[i] != 0)
        {
            a += 1;
        }
    }

    printf("a1: %d\n", a);

    for (size_t i = 0; i < 500; i++)
    {
        onnx->run(ids, atn, rsp);
    }
    

    for (size_t i = 0; i < rsp_size; i++)
    {
        if (rsp[i] != 0)
        {
            a += 1;
        }
    }

    printf("a2: %d\n", a);

    return 0;
}

// end of file
