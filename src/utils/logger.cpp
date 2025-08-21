#include "utils/logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace sv2sc::utils {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::Logger() {
    // Create console logger
    logger_ = spdlog::stdout_color_mt("sv2sc");
    logger_->set_level(spdlog::level::info);
    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
}

void Logger::setLevel(spdlog::level::level_enum level) {
    logger_->set_level(level);
}

void Logger::enableFileLogging(const std::string& filename) {
    try {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
        logger_->sinks().push_back(file_sink);
        logger_->info("File logging enabled: {}", filename);
    } catch (const spdlog::spdlog_ex& ex) {
        logger_->error("Failed to enable file logging: {}", ex.what());
    }
}

} // namespace sv2sc::utils