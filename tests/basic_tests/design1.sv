// Design1 module for VCS testing
module design1 (
    input  logic clk,
    input  logic reset,
    input  logic [7:0] in_data,
    output logic [7:0] out_data
);

    always_comb begin
        out_data = in_data ^ 8'hAA;  // Simple XOR operation
    end

endmodule