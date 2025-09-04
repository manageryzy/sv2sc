#include <catch2/catch_test_macros.hpp>
#include "codegen/systemc_generator.h"

using namespace sv2sc::codegen;

TEST_CASE("SystemC Generator Basic Functionality", "[systemc_generator]") {
    SystemCCodeGenerator generator;
    
    SECTION("Module generation") {
        generator.beginModule("test_module");
        
        // Basic functionality test - generator should accept module names
        // This tests object creation and basic method calls
        REQUIRE_NOTHROW(generator.endModule());
    }
    
    SECTION("Code generation structure") {
        generator.beginModule("counter");
        
        // Test that we can add ports using the proper API
        Port clkPort{"clk", PortDirection::INPUT, SystemCDataType::SC_LOGIC, 1, "", false, {}};
        Port countPort{"count", PortDirection::OUTPUT, SystemCDataType::SC_LV, 8, "", false, {}};
        
        REQUIRE_NOTHROW(generator.addPort(clkPort));
        REQUIRE_NOTHROW(generator.addPort(countPort));
        
        generator.endModule();
    }
}