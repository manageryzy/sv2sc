// Design module for relative path testing
module relative_design #(
    parameter WIDTH = 8
) (
    input  logic clk,
    input  logic rst,
    input  logic [WIDTH-1:0] data_in,
    output logic [WIDTH-1:0] data_out
);

always_ff @(posedge clk or posedge rst) begin
    if (rst)
        data_out <= '0;
    else
        data_out <= data_in;
end

endmodule