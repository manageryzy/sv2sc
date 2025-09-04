// Interface DUT Testbench
// Testbench for interface_dut module

module interface_tb;
    
    // Clock and reset signals
    logic clk;
    logic reset;
    
    // Interface instances
    data_if #(.WIDTH(32)) output_port();
    data_if #(.WIDTH(32)) input_port();
    ctrl_if control();
    
    // DUT instance
    interface_dut dut (
        .clk(clk),
        .output_port(output_port),
        .input_port(input_port),
        .control(control)
    );
    
    // Test control
    int test_count;
    int error_count;
    
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
        control.reset = 1;
        control.enable = 0;
        input_port.data = 32'h00000000;
        input_port.valid = 0;
        input_port.ready = 0;
        output_port.ready = 0;
        #20;
        reset = 0;
        control.reset = 0;
        #10;
        
        // Test 1: Basic data transfer
        test_count++;
        control.enable = 1;
        input_port.data = 32'hA5A5A5A5;
        input_port.valid = 1;
        #10;
        input_port.valid = 0;
        #10;
        
        if (control.busy !== 1'b1) begin
            $display("ERROR: Test %0d failed - busy signal not asserted", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Basic data transfer", test_count);
        end
        
        // Test 2: Multiple data transfers
        test_count++;
        input_port.data = 32'h5A5A5A5A;
        input_port.valid = 1;
        #10;
        input_port.data = 32'h12345678;
        #10;
        input_port.valid = 0;
        #20;
        
        if (control.done !== 1'b1) begin
            $display("ERROR: Test %0d failed - done signal not asserted", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Multiple data transfers", test_count);
        end
        
        // Test 3: Reset during operation
        test_count++;
        control.reset = 1;
        #10;
        control.reset = 0;
        #10;
        
        if (control.busy !== 1'b0 || control.done !== 1'b0) begin
            $display("ERROR: Test %0d failed - signals not reset properly", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Reset during operation", test_count);
        end
        
        // Test 4: Ready signal handling
        test_count++;
        control.enable = 1;
        output_port.ready = 1;
        input_port.data = 32'hDEADBEEF;
        input_port.valid = 1;
        #10;
        input_port.valid = 0;
        #20;
        
        if (output_port.ready !== 1'b1) begin
            $display("ERROR: Test %0d failed - ready signal not maintained", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - Ready signal handling", test_count);
        end
        
        // Test 5: Edge case - no enable
        test_count++;
        control.enable = 0;
        input_port.data = 32'hFFFFFFFF;
        input_port.valid = 1;
        #10;
        input_port.valid = 0;
        #10;
        
        if (control.busy !== 1'b0) begin
            $display("ERROR: Test %0d failed - busy signal asserted without enable", test_count);
            error_count++;
        end else begin
            $display("PASS: Test %0d - No enable condition", test_count);
        end
        
        // Wait for final result
        #20;
        
        // Print results
        $display("\n=== Interface DUT Test Results ===");
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
