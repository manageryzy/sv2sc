# Functional Tests CMakeLists.txt
# Functional testing for sv2sc core functionality

# Functional test executable
add_executable(test_sv2sc_functional test_sv2sc_functional.cpp)
target_link_libraries(test_sv2sc_functional sv2sc_core sv2sc_translator Catch2::Catch2WithMain)

# Register functional tests
add_test(NAME test_sv2sc_functional COMMAND test_sv2sc_functional)