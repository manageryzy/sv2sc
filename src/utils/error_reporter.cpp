#include "utils/error_reporter.h"
#include "utils/logger.h"
#include <iostream>
#include <sstream>
#include <fmt/format.h>

namespace sv2sc::utils {

std::string SourceLocation::toString() const {
    if (!isValid()) return "unknown location";
    
    std::string result = filename;
    if (line > 0) {
        result += ":" + std::to_string(line);
        if (column > 0) {
            result += ":" + std::to_string(column);
        }
    }
    return result;
}

void ErrorReporter::addDiagnostic(DiagnosticSeverity severity, const std::string& message,
                                 const SourceLocation& location, const std::string& suggestion,
                                 DiagnosticCategory category) {
    diagnostics_.emplace_back(severity, message, location, suggestion, category);
    
    // Also log to the regular logger
    switch (severity) {
        case DiagnosticSeverity::Info:
            LOG_INFO("{}", formatDiagnostic(diagnostics_.back()));
            break;
        case DiagnosticSeverity::Warning:
            LOG_WARN("{}", formatDiagnostic(diagnostics_.back()));
            break;
        case DiagnosticSeverity::Error:
            LOG_ERROR("{}", formatDiagnostic(diagnostics_.back()));
            break;
        case DiagnosticSeverity::Fatal:
            LOG_ERROR("FATAL: {}", formatDiagnostic(diagnostics_.back()));
            break;
    }
}

void ErrorReporter::info(const std::string& message, const SourceLocation& location) {
    addDiagnostic(DiagnosticSeverity::Info, message, location);
}

void ErrorReporter::warning(const std::string& message, const SourceLocation& location, 
                           const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Warning, message, location, suggestion);
}

void ErrorReporter::error(const std::string& message, const SourceLocation& location, 
                         const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Error, message, location, suggestion);
}

void ErrorReporter::fatal(const std::string& message, const SourceLocation& location, 
                         const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Fatal, message, location, suggestion);
}

bool ErrorReporter::hasErrors() const {
    return getErrorCount() > 0;
}

bool ErrorReporter::hasWarnings() const {
    return getWarningCount() > 0;
}

size_t ErrorReporter::getErrorCount() const {
    size_t count = 0;
    for (const auto& diag : diagnostics_) {
        if (diag.severity == DiagnosticSeverity::Error || 
            diag.severity == DiagnosticSeverity::Fatal) {
            count++;
        }
    }
    return count;
}

size_t ErrorReporter::getWarningCount() const {
    size_t count = 0;
    for (const auto& diag : diagnostics_) {
        if (diag.severity == DiagnosticSeverity::Warning) {
            count++;
        }
    }
    return count;
}

void ErrorReporter::printDiagnostics() const {
    for (const auto& diag : diagnostics_) {
        std::cout << formatDiagnostic(diag) << std::endl;
    }
}

std::string ErrorReporter::getSummary() const {
    size_t errors = getErrorCount();
    size_t warnings = getWarningCount();
    
    if (errors == 0 && warnings == 0) {
        return "Translation completed successfully with no issues.";
    }
    
    std::ostringstream oss;
    oss << "Translation completed with ";
    
    if (errors > 0) {
        oss << errors << " error" << (errors > 1 ? "s" : "");
        if (warnings > 0) {
            oss << " and ";
        }
    }
    
    if (warnings > 0) {
        oss << warnings << " warning" << (warnings > 1 ? "s" : "");
    }
    
    oss << ".";
    
    if (errors > 0) {
        oss << " Please fix the errors before using the generated code.";
    } else if (warnings > 0) {
        oss << " Please review the warnings.";
    }
    
    return oss.str();
}

void ErrorReporter::clear() {
    diagnostics_.clear();
}

void ErrorReporter::reportTranslationIssue(const std::string& message, const SourceLocation& location,
                                           const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Warning, message, location, suggestion, DiagnosticCategory::Translation);
}

void ErrorReporter::reportCompatibilityIssue(const std::string& message, const SourceLocation& location,
                                             const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Warning, message, location, suggestion, DiagnosticCategory::Compatibility);
}

void ErrorReporter::reportAdvancedFeature(const std::string& message, const SourceLocation& location,
                                          const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Info, message, location, suggestion, DiagnosticCategory::AdvancedFeatures);
}

void ErrorReporter::reportPerformanceIssue(const std::string& message, const SourceLocation& location,
                                           const std::string& suggestion) {
    addDiagnostic(DiagnosticSeverity::Info, message, location, suggestion, DiagnosticCategory::Performance);
}

std::string ErrorReporter::formatDiagnostic(const Diagnostic& diag) const {
    std::ostringstream oss;
    
    // Add color for terminal output
    oss << getSeverityColor(diag.severity);
    
    // Format: [SEVERITY] location: message
    oss << "[" << getSeverityString(diag.severity) << "]";
    
    if (diag.location.isValid()) {
        oss << " " << diag.location.toString() << ":";
    }
    
    oss << " " << diag.message;
    
    // Reset color
    oss << "\033[0m";
    
    // Add suggestion if provided
    if (!diag.suggestion.empty()) {
        oss << "\n  Suggestion: " << diag.suggestion;
    }
    
    return oss.str();
}

std::string ErrorReporter::getSeverityString(DiagnosticSeverity severity) const {
    switch (severity) {
        case DiagnosticSeverity::Info: return "INFO";
        case DiagnosticSeverity::Warning: return "WARNING";
        case DiagnosticSeverity::Error: return "ERROR";
        case DiagnosticSeverity::Fatal: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string ErrorReporter::getSeverityColor(DiagnosticSeverity severity) const {
    switch (severity) {
        case DiagnosticSeverity::Info: return "\033[36m";     // Cyan
        case DiagnosticSeverity::Warning: return "\033[33m";  // Yellow
        case DiagnosticSeverity::Error: return "\033[31m";    // Red
        case DiagnosticSeverity::Fatal: return "\033[35m";    // Magenta
        default: return "\033[0m";                            // Reset
    }
}

// Global error reporter instance
static ErrorReporter globalErrorReporter;

ErrorReporter& getGlobalErrorReporter() {
    return globalErrorReporter;
}

} // namespace sv2sc::utils
