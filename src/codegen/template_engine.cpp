#include "codegen/template_engine.h"
#include "utils/logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace sv2sc::codegen {

std::string TemplateEngine::render(const std::string& templateStr, 
                                  const VariableMap& variables,
                                  const ConditionalMap& conditionals) {
    std::string result = templateStr;
    
    // Process in order: conditionals first, then variables, then loops
    result = processConditionals(result, conditionals);
    result = processVariables(result, variables);
    result = processLoops(result, variables);
    
    return result;
}

std::string TemplateEngine::loadTemplate(const std::string& templatePath) {
    // Check cache first
    auto cacheIt = fileCache_.find(templatePath);
    if (cacheIt != fileCache_.end()) {
        LOG_DEBUG("Template loaded from cache: {}", templatePath);
        return cacheIt->second;
    }
    
    // Load from file
    std::ifstream file(templatePath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open template file: {}", templatePath);
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // Validate template syntax
    if (!validateTemplate(content)) {
        LOG_WARN("Template validation failed for: {}", templatePath);
    }
    
    // Cache the content
    fileCache_[templatePath] = content;
    LOG_DEBUG("Template loaded and cached: {}", templatePath);
    
    return content;
}

void TemplateEngine::registerTemplate(const std::string& name, const std::string& templateStr) {
    // Validate template before registering
    if (!validateTemplate(templateStr)) {
        LOG_WARN("Template validation failed for: {}", name);
    }
    
    templates_[name] = templateStr;
    LOG_DEBUG("Registered template: {}", name);
}

std::string TemplateEngine::renderTemplate(const std::string& name,
                                          const VariableMap& variables,
                                          const ConditionalMap& conditionals) {
    auto it = templates_.find(name);
    if (it == templates_.end()) {
        LOG_ERROR("Template not found: {}", name);
        return "";
    }
    
    return render(it->second, variables, conditionals);
}

void TemplateEngine::clearCache() {
    fileCache_.clear();
    LOG_DEBUG("Template file cache cleared");
}

std::string TemplateEngine::processVariables(const std::string& input, const VariableMap& variables) {
    std::string result = input;
    
    // Replace {{variable}} with values
    std::regex varRegex(R"(\{\{(\w+)\}\})");
    std::smatch match;
    
    while (std::regex_search(result, match, varRegex)) {
        std::string varName = match[1].str();
        std::string replacement = "";
        
        auto it = variables.find(varName);
        if (it != variables.end()) {
            replacement = it->second;
        } else {
            LOG_DEBUG("Variable not found in template: {}", varName);
        }
        
        result.replace(match.position(), match.length(), replacement);
    }
    
    return result;
}

std::string TemplateEngine::processConditionals(const std::string& input, const ConditionalMap& conditionals) {
    std::string result = input;
    
    // Process {{#if condition}} ... {{/if}} blocks
    std::regex ifRegex(R"(\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\})");
    std::smatch match;
    
    while (std::regex_search(result, match, ifRegex)) {
        std::string conditionName = match[1].str();
        std::string content = match[2].str();
        std::string replacement = "";
        
        auto it = conditionals.find(conditionName);
        if (it != conditionals.end() && it->second) {
            replacement = content;
        }
        
        result.replace(match.position(), match.length(), replacement);
    }
    
    // Process {{#unless condition}} ... {{/unless}} blocks
    std::regex unlessRegex(R"(\{\{#unless\s+(\w+)\}\}(.*?)\{\{/unless\}\})");
    
    while (std::regex_search(result, match, unlessRegex)) {
        std::string conditionName = match[1].str();
        std::string content = match[2].str();
        std::string replacement = "";
        
        auto it = conditionals.find(conditionName);
        if (it == conditionals.end() || !it->second) {
            replacement = content;
        }
        
        result.replace(match.position(), match.length(), replacement);
    }
    
    return result;
}

std::string TemplateEngine::processLoops(const std::string& input, const VariableMap& variables) {
    // For now, just return input - loops can be added later if needed
    return input;
}

bool TemplateEngine::validateTemplate(const std::string& templateStr) const {
    // Check for balanced conditional blocks
    std::regex ifPattern(R"(\{\{#if\s+\w+\}\})");
    std::regex endifPattern(R"(\{\{/if\}\})");
    
    auto ifBegin = std::sregex_iterator(templateStr.begin(), templateStr.end(), ifPattern);
    auto ifEnd = std::sregex_iterator();
    int ifCount = std::distance(ifBegin, ifEnd);
    
    auto endifBegin = std::sregex_iterator(templateStr.begin(), templateStr.end(), endifPattern);
    auto endifEnd = std::sregex_iterator();
    int endifCount = std::distance(endifBegin, endifEnd);
    
    if (ifCount != endifCount) {
        LOG_ERROR("Template validation failed: Unbalanced if/endif blocks ({} if, {} endif)", ifCount, endifCount);
        return false;
    }
    
    // Check for valid variable syntax
    std::regex varPattern(R"(\{\{[^#/][^}]*\}\})");
    auto varBegin = std::sregex_iterator(templateStr.begin(), templateStr.end(), varPattern);
    auto varEnd = std::sregex_iterator();
    
    for (auto it = varBegin; it != varEnd; ++it) {
        std::string var = it->str();
        // Remove {{ and }}
        std::string varName = var.substr(2, var.length() - 4);
        
        // Check for valid variable name (alphanumeric and underscore)
        if (!std::regex_match(varName, std::regex(R"([a-zA-Z_][a-zA-Z0-9_]*)"))) {
            LOG_WARN("Template validation warning: Invalid variable name: {}", varName);
        }
    }
    
    LOG_DEBUG("Template validation passed: {} if blocks, {} variables", ifCount, std::distance(varBegin, varEnd));
    return true;
}

} // namespace sv2sc::codegen
