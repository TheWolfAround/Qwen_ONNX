#include <iostream>
#include <onnxruntime_cxx_api.h>

int main()
{
    printf("greetings!\n");

    // Create an ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");

    // Get the list of available execution providers
    auto providers = Ort::GetAvailableProviders();

    // Print the available execution providers
    std::cout << "Available Execution Providers:" << std::endl;
    
    for (const auto& provider : providers)
    {
        std::cout << "- " << provider << std::endl;
    }

    return 0;
}
