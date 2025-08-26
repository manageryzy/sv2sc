#!/bin/bash
# Run comparison between SystemC and Verilator implementations

SYSTEMC_DIR=$1
VERILATOR_DIR=$2
OUTPUT_DIR=$3

if [ -z "$SYSTEMC_DIR" ] || [ -z "$VERILATOR_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <systemc_dir> <verilator_dir> <output_dir>"
    exit 1
fi

echo "========================================="
echo "PicoRV32 Implementation Comparison"
echo "========================================="
echo "SystemC dir: $SYSTEMC_DIR"
echo "Verilator dir: $VERILATOR_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Check if executables exist
if [ ! -f "$SYSTEMC_DIR/picorv32_systemc" ]; then
    echo "Error: SystemC executable not found"
    exit 1
fi

if [ ! -f "$VERILATOR_DIR/Vpicorv32" ]; then
    echo "Error: Verilator executable not found"
    exit 1
fi

# Create a simple test program if hex file doesn't exist
if [ ! -f "$OUTPUT_DIR/simple_test.hex" ]; then
    echo "Creating simple test program..."
    cat > "$OUTPUT_DIR/simple_test.hex" << 'EOF'
@00000000
00000013
00100113
00000193
00a00213
01400293
00628333
405303b3
00001437
00640023
00042483
00930663
0000a537
ead50513
00c0006f
00006537
00d50513
10000537
05000593
00b50023
04100593
00b50023
05300593
00b50023
05300593
00b50023
00a00593
00b50023
00100073
EOF
fi

# Run SystemC simulation
echo "Running SystemC simulation..."
cd "$SYSTEMC_DIR"
./picorv32_systemc --hex "$OUTPUT_DIR/simple_test.hex" --timeout 1000 > "$OUTPUT_DIR/systemc_output.txt" 2>&1
SC_RESULT=$?
echo "SystemC simulation completed with code: $SC_RESULT"

# Run Verilator simulation
echo "Running Verilator simulation..."
cd "$VERILATOR_DIR"
./Vpicorv32 --hex "$OUTPUT_DIR/simple_test.hex" --timeout 1000 > "$OUTPUT_DIR/verilator_output.txt" 2>&1
VER_RESULT=$?
echo "Verilator simulation completed with code: $VER_RESULT"

# Compare outputs
echo ""
echo "Comparing outputs..."
echo "----------------------------------------"

# Extract key information from outputs
SC_INSTRUCTIONS=$(grep -i "instructions" "$OUTPUT_DIR/systemc_output.txt" | tail -1)
VER_INSTRUCTIONS=$(grep -i "instructions" "$OUTPUT_DIR/verilator_output.txt" | tail -1)

echo "SystemC: $SC_INSTRUCTIONS"
echo "Verilator: $VER_INSTRUCTIONS"

# Check for UART output
echo ""
echo "UART Output Comparison:"
echo "SystemC UART:"
grep "UART:" "$OUTPUT_DIR/systemc_output.txt" || echo "  (no UART output)"
echo "Verilator UART:"
grep -v "Instructions\|Reset\|trap\|cycle" "$OUTPUT_DIR/verilator_output.txt" | grep -v "^$" || echo "  (no UART output)"

# Check for traps
echo ""
echo "Trap Detection:"
grep -i "trap" "$OUTPUT_DIR/systemc_output.txt" && echo "  SystemC: Trap detected" || echo "  SystemC: No trap"
grep -i "trap" "$OUTPUT_DIR/verilator_output.txt" && echo "  Verilator: Trap detected" || echo "  Verilator: No trap"

# Final result
echo ""
echo "========================================="
if [ $SC_RESULT -eq 0 ] && [ $VER_RESULT -eq 0 ]; then
    echo "RESULT: Both simulations completed successfully"
    exit 0
else
    echo "RESULT: One or both simulations failed"
    echo "  SystemC exit code: $SC_RESULT"
    echo "  Verilator exit code: $VER_RESULT"
    exit 1
fi
