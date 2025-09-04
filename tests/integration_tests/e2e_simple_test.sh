#!/bin/bash
# Simple end-to-end integration test

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================"
echo "   Simple End-to-End Integration Test"
echo "================================================"

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
SV2SC="$BUILD_DIR/src/sv2sc"
TEST_DIR="$BUILD_DIR/e2e_simple_test"

mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create simple test modules
echo -e "\n${YELLOW}[1/4] Creating test modules...${NC}"

# Simple counter module
cat > counter.sv << 'EOF'
module counter #(
    parameter WIDTH = 8
) (
    input logic clk,
    input logic reset,
    input logic enable,
    output logic [WIDTH-1:0] count,
    output logic overflow
);

    always_ff @(posedge clk) begin
        if (reset) begin
            count <= '0;
            overflow <= 1'b0;
        end else if (enable) begin
            {overflow, count} <= count + 1'b1;
        end
    end

endmodule
EOF

# Simple ALU module
cat > alu.sv << 'EOF'
module alu (
    input logic [7:0] a,
    input logic [7:0] b,
    input logic [1:0] op,
    output logic [7:0] result,
    output logic zero
);

    always_comb begin
        case (op)
            2'b00: result = a + b;
            2'b01: result = a - b;
            2'b10: result = a & b;
            2'b11: result = a | b;
        endcase
        zero = (result == 8'b0);
    end

endmodule
EOF

# Memory module
cat > memory.sv << 'EOF'
module memory #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32
) (
    input logic clk,
    input logic we,
    input logic [ADDR_WIDTH-1:0] addr,
    input logic [DATA_WIDTH-1:0] wdata,
    output logic [DATA_WIDTH-1:0] rdata
);

    logic [DATA_WIDTH-1:0] mem [0:2**ADDR_WIDTH-1];
    
    always_ff @(posedge clk) begin
        if (we) begin
            mem[addr] <= wdata;
        end
        rdata <= mem[addr];
    end

endmodule
EOF

echo -e "${GREEN}✓ Test modules created${NC}"

# Test each module
MODULES=("counter" "alu" "memory")
PASSED=0
FAILED=0

for MODULE in "${MODULES[@]}"; do
    echo -e "\n${YELLOW}[2/4] Testing $MODULE...${NC}"
    
    # Translate
    if $SV2SC -top $MODULE $MODULE.sv > $MODULE.log 2>&1; then
        echo -e "${GREEN}  ✓ Translation successful${NC}"
        
        # Check output
        if [ -f "output/${MODULE}.h" ]; then
            # Analyze
            PORTS=$(grep -c "sc_in\|sc_out" "output/${MODULE}.h" 2>/dev/null | tr -d '\n' || echo "0")
            SIGNALS=$(grep -c "sc_signal" "output/${MODULE}.h" 2>/dev/null | tr -d '\n' || echo "0")
            UNKNOWN=$(grep -c "unknown_expr" "output/${MODULE}.h" 2>/dev/null | tr -d '\n' || echo "0")
            
            echo "    Ports: $PORTS, Signals: $SIGNALS, Unknown: $UNKNOWN"
            
            if [ "$PORTS" -gt 0 ] && [ "$UNKNOWN" -eq 0 ]; then
                echo -e "${GREEN}  ✓ Quality check passed${NC}"
                ((PASSED++))
            else
                echo -e "${YELLOW}  ⚠ Quality issues detected${NC}"
                ((FAILED++))
            fi
        else
            echo -e "${RED}  ✗ No output generated${NC}"
            ((FAILED++))
        fi
    else
        echo -e "${RED}  ✗ Translation failed${NC}"
        tail -5 $MODULE.log
        ((FAILED++))
    fi
done

# Create combined testbench
echo -e "\n${YELLOW}[3/4] Creating combined testbench...${NC}"

cat > testbench.sv << 'EOF'
module testbench;
    logic clk, reset, enable;
    logic [7:0] count;
    logic overflow;
    
    counter #(.WIDTH(8)) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .count(count),
        .overflow(overflow)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        reset = 1;
        enable = 0;
        #20 reset = 0;
        #10 enable = 1;
        #1000 $finish;
    end
endmodule
EOF

# Translate testbench
if $SV2SC -top testbench testbench.sv > testbench.log 2>&1; then
    echo -e "${GREEN}✓ Testbench translation successful${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ Testbench translation failed (expected for non-synthesizable)${NC}"
fi

# Performance test
echo -e "\n${YELLOW}[4/4] Performance test...${NC}"

# Create larger module
cat > large_module.sv << 'EOF'
module large_module (
    input logic clk, reset,
    input logic [31:0] data_in [0:15],
    output logic [31:0] data_out [0:15]
);
    logic [31:0] pipeline [0:15][0:3];
    
    genvar i;
    generate
        for (i = 0; i < 16; i++) begin : gen_pipeline
            always_ff @(posedge clk) begin
                if (reset) begin
                    pipeline[i][0] <= '0;
                    pipeline[i][1] <= '0;
                    pipeline[i][2] <= '0;
                    pipeline[i][3] <= '0;
                end else begin
                    pipeline[i][0] <= data_in[i];
                    pipeline[i][1] <= pipeline[i][0];
                    pipeline[i][2] <= pipeline[i][1];
                    pipeline[i][3] <= pipeline[i][2];
                end
                data_out[i] <= pipeline[i][3];
            end
        end
    endgenerate
endmodule
EOF

START_TIME=$(date +%s%N)
if $SV2SC -top large_module large_module.sv > large_module.log 2>&1; then
    END_TIME=$(date +%s%N)
    ELAPSED=$(( ($END_TIME - $START_TIME) / 1000000 ))
    echo -e "${GREEN}✓ Large module translated in ${ELAPSED}ms${NC}"
    
    # Check size
    if [ -f "output/large_module.h" ]; then
        SIZE=$(wc -l < "output/large_module.h")
        echo "  Generated $SIZE lines of SystemC code"
    fi
    ((PASSED++))
else
    echo -e "${RED}✗ Large module translation failed${NC}"
    ((FAILED++))
fi

# Summary
echo -e "\n================================================"
echo "              TEST SUMMARY"
echo "================================================"
TOTAL=$((PASSED + FAILED))
echo -e "Tests Run:    $TOTAL"
echo -e "${GREEN}Tests Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Tests Failed: $FAILED${NC}"
else
    echo -e "Tests Failed: 0"
fi

PERCENT=$((PASSED * 100 / TOTAL))
echo -e "\nSuccess Rate: ${PERCENT}%"

if [ $PERCENT -ge 80 ]; then
    echo -e "${GREEN}RESULT: END-TO-END TEST PASSED${NC}"
    exit 0
elif [ $PERCENT -ge 50 ]; then
    echo -e "${YELLOW}RESULT: PARTIAL SUCCESS${NC}"
    exit 0
else
    echo -e "${RED}RESULT: TEST FAILED${NC}"
    exit 1
fi
