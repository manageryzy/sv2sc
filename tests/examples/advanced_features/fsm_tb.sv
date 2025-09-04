// FSM Testbench
// Comprehensive testbench for FSM DUT module

module fsm_tb;

    // Clock and reset signals
    logic clk;
    logic reset;
    
    // DUT interface signals
    logic enable;
    logic [1:0] input_signal;
    logic [7:0] data_in;
    logic [1:0] current_state;
    logic [1:0] next_state;
    logic [7:0] data_out;
    logic valid_out;
    logic busy;
    logic done;
    
    // Test control
    int test_count;
    int error_count;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // DUT instantiation
    fsm_dut #(
        .NUM_STATES(4),
        .DATA_WIDTH(8)
    ) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .input_signal(input_signal),
        .data_in(data_in),
        .current_state(current_state),
        .next_state(next_state),
        .data_out(data_out),
        .valid_out(valid_out),
        .busy(busy),
        .done(done)
    );
    
    // Test stimulus
    initial begin
        $display("Starting FSM Testbench");
        
        // Initialize signals
        reset = 1;
        enable = 0;
        input_signal = 2'b00;
        data_in = 8'h00;
        test_count = 0;
        error_count = 0;
        
        // Reset test
        $display("Test %0d: Reset Test", test_count++);
        #20;
        reset = 0;
        #10;
        
        if (current_state !== 2'b00 || !done) begin
            $display("ERROR: Reset failed - current_state=%b, done=%b", current_state, done);
            error_count++;
        end else begin
            $display("PASS: Reset test");
        end
        
        // IDLE to PROCESS transition
        $display("Test %0d: IDLE to PROCESS Transition", test_count++);
        enable = 1;
        input_signal = 2'b01; // Start processing
        data_in = 8'hA5;
        #20;
        
        if (current_state !== 2'b01 || !busy) begin
            $display("ERROR: IDLE to PROCESS failed - current_state=%b, busy=%b", current_state, busy);
            error_count++;
        end else begin
            $display("PASS: IDLE to PROCESS transition");
        end
        
        // PROCESS to WAIT transition
        $display("Test %0d: PROCESS to WAIT Transition", test_count++);
        input_signal = 2'b10; // Wait signal
        #20;
        
        if (current_state !== 2'b10) begin
            $display("ERROR: PROCESS to WAIT failed - current_state=%b", current_state);
            error_count++;
        end else begin
            $display("PASS: PROCESS to WAIT transition");
        end
        
        // WAIT to COMPLETE transition
        $display("Test %0d: WAIT to COMPLETE Transition", test_count++);
        input_signal = 2'b11; // Complete signal
        #20;
        
        if (current_state !== 2'b11 || !valid_out) begin
            $display("ERROR: WAIT to COMPLETE failed - current_state=%b, valid_out=%b", current_state, valid_out);
            error_count++;
        end else begin
            $display("PASS: WAIT to COMPLETE transition");
        end
        
        // Data processing test
        $display("Test %0d: Data Processing Test", test_count++);
        if (data_out !== 8'hA5) begin
            $display("ERROR: Data processing failed - expected data_out=A5, got data_out=%h", data_out);
            error_count++;
        end else begin
            $display("PASS: Data processing test");
        end
        
        // COMPLETE to IDLE transition
        $display("Test %0d: COMPLETE to IDLE Transition", test_count++);
        input_signal = 2'b00; // Return to idle
        #20;
        
        if (current_state !== 2'b00 || !done) begin
            $display("ERROR: COMPLETE to IDLE failed - current_state=%b, done=%b", current_state, done);
            error_count++;
        end else begin
            $display("PASS: COMPLETE to IDLE transition");
        end
        
        // Disable test
        $display("Test %0d: Disable Test", test_count++);
        enable = 0;
        input_signal = 2'b01;
        #20;
        
        if (current_state !== 2'b00) begin
            $display("ERROR: Disable failed - current_state=%b", current_state);
            error_count++;
        end else begin
            $display("PASS: Disable test");
        end
        
        // Invalid state transition test
        $display("Test %0d: Invalid State Transition Test", test_count++);
        enable = 1;
        input_signal = 2'b11; // Invalid from IDLE
        #20;
        
        if (current_state !== 2'b00) begin
            $display("ERROR: Invalid transition failed - current_state=%b", current_state);
            error_count++;
        end else begin
            $display("PASS: Invalid state transition test");
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
        $monitor("Time=%0t state=%b next=%b input=%b data_in=%h data_out=%h valid=%b busy=%b done=%b",
                 $time, current_state, next_state, input_signal, data_in, data_out, valid_out, busy, done);
    end

endmodule
