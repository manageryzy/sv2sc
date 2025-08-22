// Core module for VCS testing
module core #(
    parameter WIDTH = 32
) (
    input  logic clk,
    input  logic reset,
    input  logic [WIDTH-1:0] data_in,
    output logic [WIDTH-1:0] data_out,
    output logic valid
);

    always_ff @(posedge clk) begin
        if (reset) begin
            data_out <= '0;
            valid <= 1'b0;
        end else begin
            data_out <= data_in;
            valid <= 1'b1;
        end
    end

endmodule