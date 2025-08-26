// Error Reporting Test Module
// Tests error detection and reporting capabilities

module error_module (
    input logic clk,
    input logic reset,
    output logic [7:0] data_out
);

    // This module intentionally has some constructs that should generate warnings
    // but still translate successfully to test error reporting
    
    logic [7:0] internal_reg;
    
    // Basic functionality that should work
    always_ff @(posedge clk) begin
        if (reset) begin
            internal_reg <= 8'h00;
        end else begin
            internal_reg <= internal_reg + 1'b1;
        end
    end
    
    // Output assignment
    assign data_out = internal_reg;
    
    // Some SystemVerilog constructs that might generate warnings
    // but should still be handled gracefully
    
    // Unused signals (should generate warnings in some tools)
    logic unused_signal;
    logic [3:0] unused_vector;
    
    // Complex expressions that test expression parsing
    logic [7:0] complex_expr;
    always_comb begin
        complex_expr = internal_reg[6:0] + {internal_reg[0], 7'b0};
    end

endmodule
