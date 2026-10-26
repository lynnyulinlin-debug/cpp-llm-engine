// Engine：命令行生成示例
// 职责：解析 prompt 和 max_new_tokens，调用 Engine::generate 并打印结果。
// 运行：bash code/engine/scripts/run.sh。
// 验证：输出 finished=true 且 token 数量符合参数。
#include "cpp_llm_engine/engine.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string prompt = "hello";
    std::size_t max_new_tokens = 4;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--prompt" && index + 1 < argc) {
            options.prompt = argv[++index];
        } else if (argument == "--max-new-tokens" && index + 1 < argc) {
            options.max_new_tokens = std::stoul(argv[++index]);
        } else if (argument == "--help") {
            std::cout << "Usage: cpp_llm_engine_demo [--prompt TEXT] [--max-new-tokens N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        cpp_llm_engine::ModelConfig config;
        config.max_new_tokens = options.max_new_tokens;
        const cpp_llm_engine::Engine engine(config);
        const cpp_llm_engine::GenerationResult result = engine.generate(options.prompt);

        std::cout << "[engine] prompt: " << options.prompt << "\n";
        std::cout << "[engine] prompt tokens: " << result.prompt_tokens.size() << "\n";
        std::cout << "[engine] generated tokens: " << result.generated_tokens.size() << "\n";
        std::cout << "[engine] output: " << result.output << "\n";
        std::cout << "[engine] request finished: " << std::boolalpha << result.finished << "\n";
        return result.finished ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[engine] error: " << error.what() << "\n";
        return 1;
    }
}
