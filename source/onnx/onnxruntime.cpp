///
/// @file onnxruntime.cpp
/// @author TheWolfAround
/// @brief 
/// @version 1.0.0
/// @date 21/02/2025
/// @copyright (c) 2025 All rights reserved.
///

#include <iostream>
#include <sstream>
#include <filesystem>

#include "onnxruntime.hpp"

namespace TWA
{
    onnxruntime::onnxruntime(const std::string& model_path) noexcept
        : m_environment{nullptr},
          m_session_options{Ort::SessionOptions{}},
          m_session{nullptr},
          m_instance_name{"Qwen_2.5_0.5b_Instruct LLM Inference Driver"},
          m_memory_info{nullptr},
          m_input_node_names{"input_ids", "attention_mask", "onnx::Neg_2"},
          m_output_node_names{"logits"}
    {
        this->m_environment
            = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, m_instance_name.c_str());

        this->m_session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

        try
        {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(this->m_session_options, 0));
        }
        catch(const Ort::Exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << "CPU Execution Provider will be used.\n";
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CPU(this->m_session_options, 0));
        }

        std::filesystem::path model_fs_path = std::filesystem::u8path(model_path);
        this->m_session = Ort::Session(this->m_environment, model_fs_path.c_str(), this->m_session_options);

        this->m_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    }

    void onnxruntime::run(std::vector<int64_t> &input_ids,
                           std::vector<int64_t> &attention_mask,
                           std::vector<float16_t> &response)
    {
        this->m_input_tensors.clear();
        this->m_output_tensors.clear();

        int64_t sequence_length = static_cast<int64_t>(input_ids.size());
        std::vector<int64_t> input_dims{1, sequence_length};

        Ort::Value generated_ids
            = Ort::Value::CreateTensor<int64_t>(this->m_memory_info,
                                                input_ids.data(),
                                                sequence_length,
                                                input_dims.data(),
                                                input_dims.size());

        this->m_input_tensors.push_back(std::move(generated_ids));

        Ort::Value atn_mask
            = Ort::Value::CreateTensor<int64_t>(this->m_memory_info,
                                                attention_mask.data(),
                                                attention_mask.size(),
                                                input_dims.data(),
                                                input_dims.size());

        this->m_input_tensors.push_back(std::move(atn_mask));
        
        std::vector<int64_t> neg{0};
        std::vector<int64_t> neg_dims{1};
        Ort::Value neg_2
            = Ort::Value::CreateTensor<int64_t>(this->m_memory_info,
                                                neg.data(),
                                                neg.size(),
                                                nullptr,
                                                0);

        this->m_input_tensors.push_back(std::move(neg_2));

        std::vector<int64_t> response_dims{1, sequence_length, 151936};

        Ort::Value rsp
            = Ort::Value::CreateTensor<float16_t>(this->m_memory_info,
                                                 response.data(),
                                                 sequence_length * response_dims[2],
                                                 response_dims.data(),
                                                 response_dims.size());

        this->m_output_tensors.push_back(std::move(rsp));

        Ort::RunOptions run_options{nullptr};

        try
        {
            this->m_session.Run(run_options,
                this->m_input_node_names.data(),
                this->m_input_tensors.data(),
                this->m_input_tensors.size(),
                this->m_output_node_names.data(),
                this->m_output_tensors.data(),
                1);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }

    void onnxruntime::set_onnx_threads(const uint32_t thread_count)
    {
        this->m_session_options.SetInterOpNumThreads(2);
        this->m_session_options.SetIntraOpNumThreads(2);
    }

    void onnxruntime::print_exucution_providers()
    {
        // Get the list of available execution providers
        auto providers = Ort::GetAvailableProviders();

        // Print the available execution providers
        printf("\nAvailable Execution Providers:\n");
        
        for (const auto& provider : providers)
        {
            printf("    - %s\n", provider.c_str());
        }
    }
    
    void onnxruntime::detect_input_output_node_info()
    {
        Ort::AllocatorWithDefaultOptions allocator{};

        size_t input_node_count = this->m_session.GetInputCount();
        size_t output_node_count = this->m_session.GetOutputCount();

        std::vector<std::vector<int64_t>> input_node_dims;
        for (size_t idx = 0; idx < input_node_count; idx++)
        {
            Ort::TypeInfo input_type_info = this->m_session.GetInputTypeInfo(idx);
            Ort::ConstTensorTypeAndShapeInfo input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

            this->m_input_node_dims.push_back(input_tensor_info.GetShape());
        }
        
        std::vector<const char*> output_node_names;
        for (size_t idx = 0; idx < output_node_count; idx++)
        {
            Ort::TypeInfo output_type_info = this->m_session.GetOutputTypeInfo(idx);
            Ort::ConstTensorTypeAndShapeInfo output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            
            this->m_output_node_dims.push_back(output_tensor_info.GetShape());
        }
    }
}

// end of file
