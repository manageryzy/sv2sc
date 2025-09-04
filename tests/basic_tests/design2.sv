// Design2 module for VCS testing  
module design2 #(
    parameter DATA_WIDTH = 16
) (
    input  logic clk,
    input  logic reset,
    input  logic [DATA_WIDTH-1:0] a,
    input  logic [DATA_WIDTH-1:0] b,
    output logic [DATA_WIDTH-1:0] result
);

    always_ff @(posedge clk) begin
        if (reset)
            result <= '0;
        else
            result <= a + b;
    end

endmodule