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

#include <tokenizers_cpp.h>

static const std::string CWD = std::filesystem::current_path().string();

void tokenizers_cpp_example();

void model_inference_example();

int main()
{
    tokenizers_cpp_example();

    model_inference_example();

    return 0;
}

void tokenizers_cpp_example()
{
    ::rust::Box<::TokenizerWrapper> tokenizer = new_tokenizer("D:/Programming_Projects/Inspect/qwen_onnx_convert/Qwen2.5-0.5B-Instruct/tokenizer.json");

    ::EncodingResult result = tokenizer->encode("How are you doing! Are you alright?");

    printf("input_ids size: %zd\nattention mask size: %zd\n", result.input_ids.size(), result.attention_mask.size());

    for (size_t i = 0; i < result.input_ids.size(); i++)
    {
        printf("%d ", result.input_ids[i]);
    }
    printf("\n");

    std::vector<uint32_t> c(result.input_ids.size(), 0);

    c.assign(result.input_ids.begin(), result.input_ids.end());
    
    rust::String b = tokenizer->decode(c, false);

    printf("b: %s\n", b.c_str());
}

void model_inference_example()
{
    const std::string model_path = CWD + R"(\..\..\watanabe_onnx_float16\wtb_float16.onnx)";

    std::unique_ptr<TWA::onnxruntime> onnx{new TWA::onnxruntime(model_path)};
    onnx->set_onnx_threads(2);

    onnx->print_exucution_providers();
    onnx->detect_input_output_node_info();

    std::vector<int64_t> ids{151644, 8948, 198, 151645, 198, 151644, 872, 198, 2679, 358};
    std::vector<int64_t> atn(ids.size(), 1);
    std::vector<float16_t> rsp{};

    size_t rsp_size = ids.size() * 151936;

    rsp.resize(rsp_size);

    int a = 0;

    for (size_t i = 0; i < rsp_size; i++)
    {
        if (rsp[i].ToFloat() != 0)
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
        if (rsp[i].ToFloat() != 0)
        {
            a += 1;
        }
    }

    printf("a2: %d\n", a);
}

// end of file
