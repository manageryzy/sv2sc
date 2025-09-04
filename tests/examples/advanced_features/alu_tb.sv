// ALU Testbench
// Comprehensive testbench for ALU DUT module

module alu_tb;

    // Clock and reset signals
    logic clk;
    logic reset;
    
    // DUT interface signals
    logic enable;
    logic [3:0] opcode;
    logic [31:0] operand_a;
    logic [31:0] operand_b;
    logic [4:0] shift_amount;
    logic [31:0] result;
    logic zero_flag;
    logic carry_flag;
    logic overflow_flag;
    logic negative_flag;
    logic ready;
    
    // Test control
    int test_count;
    int error_count;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // DUT instantiation
    alu_dut #(
        .WIDTH(32)
    ) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .opcode(opcode),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .shift_amount(shift_amount),
        .result(result),
        .zero_flag(zero_flag),
        .carry_flag(carry_flag),
        .overflow_flag(overflow_flag),
        .negative_flag(negative_flag),
        .ready(ready)
    );
    
    // Test stimulus
    initial begin
        $display("Starting ALU Testbench");
        
        // Initialize signals
        reset = 1;
        enable = 0;
        opcode = 4'h0;
        operand_a = 32'h0;
        operand_b = 32'h0;
        shift_amount = 5'h0;
        test_count = 0;
        error_count = 0;
        
        // Reset test
        $display("Test %0d: Reset Test", test_count++);
        #20;
        reset = 0;
        #10;
        
        if (result !== 32'h0 || !ready) begin
            $display("ERROR: Reset failed - result=%h, ready=%b", result, ready);
            error_count++;
        end else begin
            $display("PASS: Reset test");
        end
        
        // ADD test
        $display("Test %0d: ADD Test", test_count++);
        enable = 1;
        opcode = 4'h0; // ADD
        operand_a = 32'h0000000A;
        operand_b = 32'h00000005;
        #20;
        
        if (result !== 32'h0000000F) begin
            $display("ERROR: ADD failed - expected result=0F, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: ADD test");
        end
        
        // SUB test
        $display("Test %0d: SUB Test", test_count++);
        opcode = 4'h1; // SUB
        operand_a = 32'h0000000F;
        operand_b = 32'h00000005;
        #20;
        
        if (result !== 32'h0000000A) begin
            $display("ERROR: SUB failed - expected result=0A, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: SUB test");
        end
        
        // AND test
        $display("Test %0d: AND Test", test_count++);
        opcode = 4'h2; // AND
        operand_a = 32'h0000000F;
        operand_b = 32'h0000000A;
        #20;
        
        if (result !== 32'h0000000A) begin
            $display("ERROR: AND failed - expected result=0A, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: AND test");
        end
        
        // OR test
        $display("Test %0d: OR Test", test_count++);
        opcode = 4'h3; // OR
        operand_a = 32'h0000000F;
        operand_b = 32'h0000000A;
        #20;
        
        if (result !== 32'h0000000F) begin
            $display("ERROR: OR failed - expected result=0F, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: OR test");
        end
        
        // XOR test
        $display("Test %0d: XOR Test", test_count++);
        opcode = 4'h4; // XOR
        operand_a = 32'h0000000F;
        operand_b = 32'h0000000A;
        #20;
        
        if (result !== 32'h00000005) begin
            $display("ERROR: XOR failed - expected result=05, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: XOR test");
        end
        
        // NOT test
        $display("Test %0d: NOT Test", test_count++);
        opcode = 4'h5; // NOT
        operand_a = 32'h0000000F;
        #20;
        
        if (result !== 32'hFFFFFFF0) begin
            $display("ERROR: NOT failed - expected result=FFFFFFF0, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: NOT test");
        end
        
        // SLL test
        $display("Test %0d: SLL Test", test_count++);
        opcode = 4'h6; // SLL
        operand_a = 32'h0000000F;
        shift_amount = 5'h02;
        #20;
        
        if (result !== 32'h0000003C) begin
            $display("ERROR: SLL failed - expected result=3C, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: SLL test");
        end
        
        // SRL test
        $display("Test %0d: SRL Test", test_count++);
        opcode = 4'h7; // SRL
        operand_a = 32'h0000003C;
        shift_amount = 5'h02;
        #20;
        
        if (result !== 32'h0000000F) begin
            $display("ERROR: SRL failed - expected result=0F, got result=%h", result);
            error_count++;
        end else begin
            $display("PASS: SRL test");
        end
        
        // Zero flag test
        $display("Test %0d: Zero Flag Test", test_count++);
        opcode = 4'h0; // ADD
        operand_a = 32'h00000000;
        operand_b = 32'h00000000;
        #20;
        
        if (!zero_flag) begin
            $display("ERROR: Zero flag failed - zero_flag=%b", zero_flag);
            error_count++;
        end else begin
            $display("PASS: Zero flag test");
        end
        
        // Negative flag test
        $display("Test %0d: Negative Flag Test", test_count++);
        opcode = 4'h1; // SUB
        operand_a = 32'h00000005;
        operand_b = 32'h0000000F;
        #20;
        
        if (!negative_flag) begin
            $display("ERROR: Negative flag failed - negative_flag=%b", negative_flag);
            error_count++;
        end else begin
            $display("PASS: Negative flag test");
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
        $monitor("Time=%0t opcode=%h a=%h b=%h result=%h zero=%b neg=%b ready=%b",
                 $time, opcode, operand_a, operand_b, result, zero_flag, negative_flag, ready);
    end

endmodule
