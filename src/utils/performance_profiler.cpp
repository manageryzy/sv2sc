#include "utils/performance_profiler.h"
#include "utils/logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

namespace sv2sc::utils {

ScopedProfiler::ScopedProfiler(const std::string& name) {
    if (getGlobalProfiler().isEnabled()) {
        point_ = std::make_shared<ProfilePoint>(name);
        point_->start();
    }
}

ScopedProfiler::~ScopedProfiler() {
    if (point_) {
        point_->end();
        getGlobalProfiler().addProfilePoint(point_);
    }
}

void PerformanceProfiler::startProfile(const std::string& name) {
    if (!enabled_) return;
    
    auto point = std::make_shared<ProfilePoint>(name);
    point->start();
    activeProfiles_[name] = point;
}

void PerformanceProfiler::endProfile(const std::string& name) {
    if (!enabled_) return;
    
    auto it = activeProfiles_.find(name);
    if (it != activeProfiles_.end()) {
        it->second->end();
        completedProfiles_.push_back(it->second);
        activeProfiles_.erase(it);
    }
}

void PerformanceProfiler::addProfilePoint(std::shared_ptr<ProfilePoint> point) {
    if (!enabled_ || !point || !point->isComplete()) return;
    
    completedProfiles_.push_back(point);
}

double PerformanceProfiler::getTotalTime() const {
    double total = 0.0;
    for (const auto& point : completedProfiles_) {
        total += point->durationMs;
    }
    return total;
}

double PerformanceProfiler::getOperationTime(const std::string& name) const {
    double total = 0.0;
    for (const auto& point : completedProfiles_) {
        if (point->name == name) {
            total += point->durationMs;
        }
    }
    return total;
}

void PerformanceProfiler::printSummary() const {
    if (!enabled_ || completedProfiles_.empty()) {
        std::cout << "No performance data available.\n";
        return;
    }
    
    std::cout << getReport() << std::endl;
}

std::string PerformanceProfiler::getReport() const {
    if (!enabled_ || completedProfiles_.empty()) {
        return "No performance data available.";
    }
    
    std::ostringstream oss;
    oss << "\n=== Performance Report ===\n";
    
    // Group operations by name and calculate totals
    std::unordered_map<std::string, double> operationTotals;
    std::unordered_map<std::string, int> operationCounts;
    
    for (const auto& point : completedProfiles_) {
        operationTotals[point->name] += point->durationMs;
        operationCounts[point->name]++;
    }
    
    // Sort by total time (descending)
    std::vector<std::pair<std::string, double>> sortedOps;
    for (const auto& op : operationTotals) {
        sortedOps.emplace_back(op.first, op.second);
    }
    std::sort(sortedOps.begin(), sortedOps.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Print header
    oss << std::left << std::setw(30) << "Operation" 
        << std::right << std::setw(12) << "Total Time" 
        << std::setw(8) << "Count" 
        << std::setw(12) << "Avg Time" << "\n";
    oss << std::string(62, '-') << "\n";
    
    // Print operations
    double totalTime = getTotalTime();
    for (const auto& op : sortedOps) {
        const std::string& name = op.first;
        double total = op.second;
        int count = operationCounts[name];
        double avg = total / count;
        double percentage = (total / totalTime) * 100.0;
        
        oss << std::left << std::setw(30) << name
            << std::right << std::setw(10) << formatDuration(total)
            << " (" << std::setw(4) << std::fixed << std::setprecision(1) << percentage << "%)"
            << std::setw(6) << count
            << std::setw(12) << formatDuration(avg) << "\n";
    }
    
    oss << std::string(62, '-') << "\n";
    oss << std::left << std::setw(30) << "TOTAL"
        << std::right << std::setw(12) << formatDuration(totalTime) << "\n";
    
    return oss.str();
}

void PerformanceProfiler::clear() {
    activeProfiles_.clear();
    completedProfiles_.clear();
}

std::string PerformanceProfiler::formatDuration(double ms) const {
    if (ms < 1.0) {
        return fmt::format("{:.2f}μs", ms * 1000.0);
    } else if (ms < 1000.0) {
        return fmt::format("{:.2f}ms", ms);
    } else {
        return fmt::format("{:.2f}s", ms / 1000.0);
    }
}

// Global profiler instance
static PerformanceProfiler globalProfiler;

PerformanceProfiler& getGlobalProfiler() {
    return globalProfiler;
}

} // namespace sv2sc::utils
