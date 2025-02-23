///
/// @file onnxruntime.hpp
/// @author TheWolfAround
/// @brief 
/// @version 1.0.0
/// @date 21/02/2025
/// @copyright (c) 2025 All rights reserved.
///

#ifndef TWA_ONNXRUNTIME_HPP
#define TWA_ONNXRUNTIME_HPP

#include <string>
#include <typeindex>

#include <onnxruntime_cxx_api.h>
#include <dml/dml_provider_factory.h>
#include <cpu/cpu_provider_factory.h>

///
/// @namespace TWA
/// @brief TheWolfAround Namespace
///
namespace TWA
{
    /// 
    /// @class onnxruntime
    /// @brief 
    ///
    class onnxruntime
    {
    public:
    ////////////////////////////////////////////////////////////////////////////
    // Special Members
    ////////////////////////////////////////////////////////////////////////////

        onnxruntime() noexcept = delete; /**< Deleted default constructor */
        ~onnxruntime() noexcept = default; /**< Default destructor */
        onnxruntime(onnxruntime &&) noexcept = delete; /**< Deleted move constructor */
        onnxruntime &operator=(onnxruntime &&) noexcept = delete; /**< Deleted move assignment operator */
        onnxruntime(const onnxruntime &) noexcept = delete; /**< Deleted default copy constructor */
        onnxruntime &operator=(onnxruntime &) noexcept = delete; /**< Deleted default copy assignment operator */

    ////////////////////////////////////////////////////////////////////////////
    // Public Members
    ////////////////////////////////////////////////////////////////////////////

        onnxruntime(const std::string& model_path) noexcept; /**< Default constructor */

        void run(std::vector<int64_t>& input_ids,
                  std::vector<int64_t>& attention_mask,
                  std::vector<float>& response);

        void set_onnx_threads(uint32_t thread_count);

        void print_exucution_providers();

        void detect_input_output_node_info();

        void print_input_output_node_info();

    ////////////////////////////////////////////////////////////////////////////
    // Private Members
    ////////////////////////////////////////////////////////////////////////////
    private:

        void create_tensor();

        Ort::Env m_environment;

        Ort::SessionOptions m_session_options;

        Ort::Session m_session;

        std::string m_instance_name;

        const std::vector<const char*> m_input_node_names;
        const std::vector<const char*> m_output_node_names;

        std::vector<std::vector<int64_t>> m_input_node_dims;
        std::vector<std::vector<int64_t>> m_output_node_dims;

        Ort::MemoryInfo m_memory_info;

        std::vector<Ort::Value> m_input_tensors;

        std::vector<Ort::Value> m_output_tensors;

    ////////////////////////////////////////////////////////////////////////////
    // Protected Members
    ////////////////////////////////////////////////////////////////////////////
    protected:
    }; // onnxruntime

} // TWA

#endif // TWA_ONNXRUNTIME_HPP

// end of file
