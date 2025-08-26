#ifndef SV2SC_PERFORMANCE_PROFILER_H
#define SV2SC_PERFORMANCE_PROFILER_H

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace sv2sc::utils {

/**
 * @brief Performance measurement point
 */
struct ProfilePoint {
    std::string name;
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    double durationMs = 0.0;
    
    ProfilePoint(const std::string& n) : name(n) {}
    
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    void end() {
        endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        durationMs = duration.count() / 1000.0;
    }
    
    bool isComplete() const {
        return durationMs > 0.0;
    }
};

/**
 * @brief RAII-style profiler for automatic timing
 */
class ScopedProfiler {
public:
    ScopedProfiler(const std::string& name);
    ~ScopedProfiler();
    
private:
    std::shared_ptr<ProfilePoint> point_;
};

/**
 * @brief Performance profiler for measuring translation performance
 */
class PerformanceProfiler {
public:
    /**
     * @brief Start profiling a named operation
     */
    void startProfile(const std::string& name);
    
    /**
     * @brief End profiling a named operation
     */
    void endProfile(const std::string& name);
    
    /**
     * @brief Add a completed profile point
     */
    void addProfilePoint(std::shared_ptr<ProfilePoint> point);
    
    /**
     * @brief Get total elapsed time for all operations
     */
    double getTotalTime() const;
    
    /**
     * @brief Get elapsed time for a specific operation
     */
    double getOperationTime(const std::string& name) const;
    
    /**
     * @brief Print performance summary
     */
    void printSummary() const;
    
    /**
     * @brief Get formatted performance report
     */
    std::string getReport() const;
    
    /**
     * @brief Clear all profile data
     */
    void clear();
    
    /**
     * @brief Check if profiling is enabled
     */
    bool isEnabled() const { return enabled_; }
    
    /**
     * @brief Enable/disable profiling
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }

private:
    bool enabled_ = true;
    std::unordered_map<std::string, std::shared_ptr<ProfilePoint>> activeProfiles_;
    std::vector<std::shared_ptr<ProfilePoint>> completedProfiles_;
    
    /**
     * @brief Format duration for display
     */
    std::string formatDuration(double ms) const;
};

/**
 * @brief Global performance profiler instance
 */
extern PerformanceProfiler& getGlobalProfiler();

/**
 * @brief Convenience macros for profiling
 */
#define PROFILE_SCOPE(name) \
    sv2sc::utils::ScopedProfiler _prof(name)

#define PROFILE_START(name) \
    sv2sc::utils::getGlobalProfiler().startProfile(name)

#define PROFILE_END(name) \
    sv2sc::utils::getGlobalProfiler().endProfile(name)

} // namespace sv2sc::utils

#endif // SV2SC_PERFORMANCE_PROFILER_H
