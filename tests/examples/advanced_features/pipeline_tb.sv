// Pipeline DUT Testbench
// Testbench for pipeline_dut module

module pipeline_tb;
    
    // Clock and reset signals
    logic clk;
    logic reset;
    
    // DUT interface signals
    logic enable;
    logic [31:0] data_in;
    logic valid_in;
    logic [31:0] data_out;
    logic valid_out;
    logic [3:0] stage_valid;
    logic busy;
    logic done;
    
    // Test control
    int test_count;
    int error_count;
    
    // Instantiate DUT
    pipeline_dut dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .data_in(data_in),
        .valid_in(valid_in),
        .data_out(data_out),
        .valid_out(valid_out),
        .stage_valid(stage_valid),
        .busy(busy),
        .done(done)
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
        enable = 0;
        data_in = 32'h00000000;
        valid_in = 0;
        #20;
        reset = 0;
        #10;
        
        // Test 1: Basic pipeline operation
        test_count++;
        enable = 1;
        data_in = 32'hA5A5A5A5;
        valid_in = 1;
        #10;
        valid_in = 0;
        #40; // Wait for pipeline to fill
        
        if (valid_out !== 1'b1) begin
            $display("ERROR: Test %0d failed - valid_out not asserted", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Basic pipeline operation", test_count);
        end
        
        // Test 2: Multiple data items
        test_count++;
        data_in = 32'h12345678;
        valid_in = 1;
        #10;
        data_in = 32'h87654321;
        #10;
        data_in = 32'hDEADBEEF;
        #10;
        valid_in = 0;
        #50; // Wait for all data to propagate
        
        if (busy !== 1'b1) begin
            $display("ERROR: Test %0d failed - busy signal not asserted", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Multiple data items", test_count);
        end
        
        // Test 3: Pipeline stages
        test_count++;
        enable = 1;
        data_in = 32'h11111111;
        valid_in = 1;
        #10;
        valid_in = 0;
        #10;
        
        if (stage_valid !== 4'b0001) begin
            $display("ERROR: Test %0d failed - stage_valid incorrect: %b", test_count, stage_valid);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Pipeline stages", test_count);
        end
        
        #30; // Wait for more stages
        
        // Test 4: Reset during operation
        test_count++;
        reset = 1;
        #10;
        reset = 0;
        #10;
        
        if (busy !== 1'b0 || done !== 1'b0 || valid_out !== 1'b0) begin
            $display("ERROR: Test %0d failed - signals not reset properly", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Reset during operation", test_count);
        end
        
        // Test 5: Disable during operation
        test_count++;
        enable = 1;
        data_in = 32'h22222222;
        valid_in = 1;
        #10;
        valid_in = 0;
        enable = 0;
        #20;
        
        if (busy !== 1'b0) begin
            $display("ERROR: Test %0d failed - busy signal not deasserted when disabled", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Disable during operation", test_count);
        end
        
        // Test 6: Pipeline completion
        test_count++;
        enable = 1;
        data_in = 32'h33333333;
        valid_in = 1;
        #10;
        valid_in = 0;
        #50; // Wait for completion
        
        if (done !== 1'b1) begin
            $display("ERROR: Test %0d failed - done signal not asserted", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Pipeline completion", test_count);
        end
        
        // Wait for final result
        #20;
        
        // Print results
        $display("\n=== Pipeline DUT Test Results ===");
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
