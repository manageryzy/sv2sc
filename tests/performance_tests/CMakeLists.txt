# Performance Tests CMakeLists.txt
# Performance benchmarking for sv2sc components

# NBA splitting benchmark
add_executable(nba_splitting_benchmark nba_splitting_benchmark.cpp)
target_link_libraries(nba_splitting_benchmark sv2sc_core sv2sc_translator)

# Register performance tests
add_test(NAME nba_splitting_benchmark COMMAND nba_splitting_benchmark)