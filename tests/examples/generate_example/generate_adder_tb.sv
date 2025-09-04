// Generate Adder Testbench
// Testbench for generate_adder DUT module

module generate_adder_tb;
    
    // Clock and reset signals
    logic clk;
    logic reset;
    
    // DUT interface signals
    logic [7:0] a;
    logic [7:0] b;
    logic [8:0] sum;
    
    // Test control
    int test_count;
    int error_count;
    
    // Instantiate DUT
    generate_adder dut (
        .clk(clk),
        .reset(reset),
        .a(a),
        .b(b),
        .sum(sum)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test sequence
    initial begin
        // Initialize
        test_count = 0;
        error_count = 0;
        
        // Reset sequence
        reset = 1;
        a = 8'h00;
        b = 8'h00;
        #20;
        reset = 0;
        #10;
        
        // Test 1: Basic addition
        test_count++;
        a = 8'h05;
        b = 8'h03;
        #10;
        if (sum !== 9'h08) begin
            $display("ERROR: Test %0d failed - expected 8, got %0h", test_count, sum);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Basic addition", test_count);
        end
        
        // Test 2: Zero addition
        test_count++;
        a = 8'h00;
        b = 8'h00;
        #10;
        if (sum !== 9'h00) begin
            $display("ERROR: Test %0d failed - expected 0, got %0h", test_count, sum);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Zero addition", test_count);
        end
        
        // Test 3: Maximum values
        test_count++;
        a = 8'hFF;
        b = 8'hFF;
        #10;
        if (sum !== 9'h1FE) begin
            $display("ERROR: Test %0d failed - expected 1FE, got %0h", test_count, sum);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Maximum values", test_count);
        end
        
        // Test 4: Random values
        test_count++;
        a = 8'hA5;
        b = 8'h5A;
        #10;
        if (sum !== 9'h0FF) begin
            $display("ERROR: Test %0d failed - expected FF, got %0h", test_count, sum);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Random values", test_count);
        end
        
        // Test 5: Carry test
        test_count++;
        a = 8'h80;
        b = 8'h80;
        #10;
        if (sum !== 9'h100) begin
            $display("ERROR: Test %0d failed - expected 100, got %0h", test_count, sum);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Carry test", test_count);
        end
        
        // Wait for final result
        #20;
        
        // Print results
        $display("\n=== Generate Adder Test Results ===");
        $display("Total Tests: %0d", test_count);
        $display("Errors: %0d", error_count);
        
        if (error_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $finish;
    end
    
endmodule
