#ifndef SV2SC_TEMPLATE_ENGINE_H
#define SV2SC_TEMPLATE_ENGINE_H

#include <string>
#include <unordered_map>
#include <vector>
#include <regex>
#include <functional>

namespace sv2sc::codegen {

/**
 * @brief Simple template engine for SystemC code generation
 * 
 * Supports variable substitution with {{variable}} syntax
 * and conditional blocks with {{#if condition}} {{/if}} syntax
 */
class TemplateEngine {
public:
    using VariableMap = std::unordered_map<std::string, std::string>;
    using ConditionalMap = std::unordered_map<std::string, bool>;
    
    /**
     * @brief Render a template with given variables and conditionals
     */
    std::string render(const std::string& templateStr, 
                      const VariableMap& variables = {},
                      const ConditionalMap& conditionals = {});
    
    /**
     * @brief Load template from file
     */
    std::string loadTemplate(const std::string& templatePath);
    
    /**
     * @brief Register a template string with a name
     */
    void registerTemplate(const std::string& name, const std::string& templateStr);
    
    /**
     * @brief Render a registered template
     */
    std::string renderTemplate(const std::string& name,
                              const VariableMap& variables = {},
                              const ConditionalMap& conditionals = {});
    
    /**
     * @brief Clear file cache to free memory
     */
    void clearCache();

private:
    std::unordered_map<std::string, std::string> templates_;
    std::unordered_map<std::string, std::string> fileCache_;  // Cache for loaded template files
    
    /**
     * @brief Process variable substitutions
     */
    std::string processVariables(const std::string& input, const VariableMap& variables);
    
    /**
     * @brief Process conditional blocks
     */
    std::string processConditionals(const std::string& input, const ConditionalMap& conditionals);
    
    /**
     * @brief Process loops (for future extension)
     */
    std::string processLoops(const std::string& input, const VariableMap& variables);
    
    /**
     * @brief Validate template syntax
     */
    bool validateTemplate(const std::string& templateStr) const;
};

} // namespace sv2sc::codegen

#endif // SV2SC_TEMPLATE_ENGINE_H
