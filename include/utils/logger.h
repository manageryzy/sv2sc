#pragma once

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <memory>

namespace sv2sc::utils {

class Logger {
public:
    static Logger& getInstance();
    
    void setLevel(spdlog::level::level_enum level);
    void enableFileLogging(const std::string& filename);
    
    template<typename... Args>
    void debug(const std::string& format, Args&&... args) {
        logger_->debug(fmt::runtime(format), std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void info(const std::string& format, Args&&... args) {
        logger_->info(fmt::runtime(format), std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void warn(const std::string& format, Args&&... args) {
        logger_->warn(fmt::runtime(format), std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void error(const std::string& format, Args&&... args) {
        logger_->error(fmt::runtime(format), std::forward<Args>(args)...);
    }

private:
    Logger();
    std::shared_ptr<spdlog::logger> logger_;
};

#define LOG_DEBUG(...) sv2sc::utils::Logger::getInstance().debug(__VA_ARGS__)
#define LOG_INFO(...) sv2sc::utils::Logger::getInstance().info(__VA_ARGS__)
#define LOG_WARN(...) sv2sc::utils::Logger::getInstance().warn(__VA_ARGS__)
#define LOG_ERROR(...) sv2sc::utils::Logger::getInstance().error(__VA_ARGS__)

} // namespace sv2sc::utils