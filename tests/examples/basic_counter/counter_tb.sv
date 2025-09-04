// Counter Testbench
// Comprehensive testbench for counter DUT module

module counter_tb;

    // Clock and reset signals
    logic clk;
    logic reset;
    
    // DUT interface signals
    logic enable;
    logic load;
    logic [7:0] load_value;
    logic [7:0] count;
    logic overflow;
    
    // Test control
    int test_count;
    int error_count;
    logic [7:0] count_before;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // DUT instantiation
    counter #(
        .WIDTH(8),
        .MAX_COUNT(255)
    ) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .load(load),
        .load_value(load_value),
        .count(count),
        .overflow(overflow)
    );
    
    // Test stimulus
    initial begin
        $display("Starting Counter Testbench");
        
        // Initialize signals
        reset = 1;
        enable = 0;
        load = 0;
        load_value = 8'h00;
        test_count = 0;
        error_count = 0;
        
        // Reset test
        $display("Test %0d: Reset Test", test_count++);
        #20;
        reset = 0;
        #10;
        
        if (count !== 8'h00 || overflow !== 1'b0) begin
            $display("ERROR: Reset failed - count=%h, overflow=%b", count, overflow);
            error_count++;
        end else begin
            $display("PASS: Reset test");
        end
        
        // Enable test
        $display("Test %0d: Enable Test", test_count++);
        enable = 1;
        #50; // 5 clock cycles
        
        if (count !== 8'h05) begin
            $display("ERROR: Enable failed - expected count=5, got count=%h", count);
            error_count++;
        end else begin
            $display("PASS: Enable test");
        end
        
        // Load test
        $display("Test %0d: Load Test", test_count++);
        load = 1;
        load_value = 8'hA5;
        #10;
        load = 0;
        #10;
        
        if (count !== 8'hA5) begin
            $display("ERROR: Load failed - expected count=A5, got count=%h", count);
            error_count++;
        end else begin
            $display("PASS: Load test");
        end
        
        // Overflow test
        $display("Test %0d: Overflow Test", test_count++);
        load_value = 8'hFE;
        load = 1;
        #10;
        load = 0;
        #20; // 2 clock cycles to reach 255
        
        if (count !== 8'h00 || overflow !== 1'b1) begin
            $display("ERROR: Overflow failed - count=%h, overflow=%b", count, overflow);
            error_count++;
        end else begin
            $display("PASS: Overflow test");
        end
        
        // Disable test
        $display("Test %0d: Disable Test", test_count++);
        enable = 0;
        #20;
        count_before = count;
        #20;
        
        if (count !== count_before) begin
            $display("ERROR: Disable failed - count changed from %h to %h", count_before, count);
            error_count++;
        end else begin
            $display("PASS: Disable test");
        end
        
        // Final results
        $display("Testbench completed: %0d tests, %0d errors", test_count, error_count);
        if (error_count == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("SOME TESTS FAILED!");
        end
        
        $finish;
    end
    
    // Monitor for debugging
    initial begin
        $monitor("Time=%0t clk=%b reset=%b enable=%b load=%b load_value=%h count=%h overflow=%b",
                 $time, clk, reset, enable, load, load_value, count, overflow);
    end

endmodule
