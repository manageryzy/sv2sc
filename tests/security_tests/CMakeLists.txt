# Security Tests CMakeLists.txt
# Security validation for sv2sc components

# Security validation test
add_executable(test_security_validation test_security_validation.cpp)
target_link_libraries(test_security_validation sv2sc_core Catch2::Catch2WithMain)

# Register security tests
add_test(NAME test_security_validation COMMAND test_security_validation)