#ifndef SV2SC_ERROR_REPORTER_H
#define SV2SC_ERROR_REPORTER_H

#include <string>
#include <vector>
#include <memory>

namespace sv2sc::utils {

/**
 * @brief Severity levels for diagnostic messages
 */
enum class DiagnosticSeverity {
    Info,
    Warning,
    Error,
    Fatal
};

/**
 * @brief Categories for different types of diagnostics
 */
enum class DiagnosticCategory {
    General,
    Syntax,
    Translation,
    Performance,
    Compatibility,
    AdvancedFeatures
};

/**
 * @brief Location information for diagnostics
 */
struct SourceLocation {
    std::string filename;
    int line = 0;
    int column = 0;
    
    SourceLocation() = default;
    SourceLocation(const std::string& file, int ln = 0, int col = 0) 
        : filename(file), line(ln), column(col) {}
    
    bool isValid() const { return !filename.empty() && line > 0; }
    std::string toString() const;
};

/**
 * @brief A diagnostic message with location and severity
 */
struct Diagnostic {
    DiagnosticSeverity severity;
    DiagnosticCategory category;
    std::string message;
    SourceLocation location;
    std::string suggestion;  // Optional suggestion for fixing the issue
    
    Diagnostic(DiagnosticSeverity sev, const std::string& msg, 
               const SourceLocation& loc = {}, const std::string& hint = "",
               DiagnosticCategory cat = DiagnosticCategory::General)
        : severity(sev), category(cat), message(msg), location(loc), suggestion(hint) {}
};

/**
 * @brief Enhanced error reporting system with diagnostics and suggestions
 */
class ErrorReporter {
public:
    /**
     * @brief Add a diagnostic message
     */
    void addDiagnostic(DiagnosticSeverity severity, const std::string& message,
                      const SourceLocation& location = {}, const std::string& suggestion = "",
                      DiagnosticCategory category = DiagnosticCategory::General);
    
    /**
     * @brief Convenience methods for different severity levels
     */
    void info(const std::string& message, const SourceLocation& location = {});
    void warning(const std::string& message, const SourceLocation& location = {}, 
                const std::string& suggestion = "");
    void error(const std::string& message, const SourceLocation& location = {}, 
              const std::string& suggestion = "");
    void fatal(const std::string& message, const SourceLocation& location = {}, 
              const std::string& suggestion = "");
    
    /**
     * @brief Check if there are any errors
     */
    bool hasErrors() const;
    bool hasWarnings() const;
    
    /**
     * @brief Get diagnostic counts
     */
    size_t getErrorCount() const;
    size_t getWarningCount() const;
    size_t getDiagnosticCount() const { return diagnostics_.size(); }
    
    /**
     * @brief Print all diagnostics to console
     */
    void printDiagnostics() const;
    
    /**
     * @brief Get formatted diagnostic summary
     */
    std::string getSummary() const;
    
    /**
     * @brief Clear all diagnostics
     */
    void clear();
    
    /**
     * @brief Get all diagnostics
     */
    const std::vector<Diagnostic>& getDiagnostics() const { return diagnostics_; }
    
    /**
     * @brief Category-specific convenience methods
     */
    void reportTranslationIssue(const std::string& message, const SourceLocation& location = {},
                                const std::string& suggestion = "");
    void reportCompatibilityIssue(const std::string& message, const SourceLocation& location = {},
                                  const std::string& suggestion = "");
    void reportAdvancedFeature(const std::string& message, const SourceLocation& location = {},
                               const std::string& suggestion = "");
    void reportPerformanceIssue(const std::string& message, const SourceLocation& location = {},
                                const std::string& suggestion = "");

private:
    std::vector<Diagnostic> diagnostics_;
    
    /**
     * @brief Format a diagnostic for display
     */
    std::string formatDiagnostic(const Diagnostic& diag) const;
    
    /**
     * @brief Get severity string
     */
    std::string getSeverityString(DiagnosticSeverity severity) const;
    
    /**
     * @brief Get severity color code (for terminal output)
     */
    std::string getSeverityColor(DiagnosticSeverity severity) const;
};

/**
 * @brief Global error reporter instance
 */
extern ErrorReporter& getGlobalErrorReporter();

/**
 * @brief Common error reporting macros for convenience
 */
#define REPORT_INFO(msg, ...) \
    sv2sc::utils::getGlobalErrorReporter().info(fmt::format(msg, ##__VA_ARGS__))

#define REPORT_WARNING(msg, loc, suggestion, ...) \
    sv2sc::utils::getGlobalErrorReporter().warning(fmt::format(msg, ##__VA_ARGS__), loc, suggestion)

#define REPORT_ERROR(msg, loc, suggestion, ...) \
    sv2sc::utils::getGlobalErrorReporter().error(fmt::format(msg, ##__VA_ARGS__), loc, suggestion)

#define REPORT_FATAL(msg, loc, suggestion, ...) \
    sv2sc::utils::getGlobalErrorReporter().fatal(fmt::format(msg, ##__VA_ARGS__), loc, suggestion)

} // namespace sv2sc::utils

#endif // SV2SC_ERROR_REPORTER_H
