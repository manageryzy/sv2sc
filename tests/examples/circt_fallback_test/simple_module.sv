// Simple module to test CIRCT fallback generator
module simple_module (
    input logic clk,
    input logic reset,
    input logic [7:0] data_in,
    output logic [7:0] data_out,
    output logic valid
);

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            data_out <= 8'h00;
            valid <= 1'b0;
        end else begin
            data_out <= data_in;
            valid <= 1'b1;
        end
    end

endmodule

