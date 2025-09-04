// Basic Counter DUT Module
// Tests basic SystemVerilog constructs and translation to SystemC

module counter #(
    parameter WIDTH = 8,
    parameter MAX_COUNT = 255
)(
    input  logic clk,
    input  logic reset,
    input  logic enable,
    input  logic load,
    input  logic [WIDTH-1:0] load_value,
    output logic [WIDTH-1:0] count,
    output logic overflow
);

    logic [WIDTH-1:0] count_reg;
    logic overflow_reg;

    // Sequential logic for counter
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            count_reg <= '0;
            overflow_reg <= 1'b0;
        end else if (load) begin
            count_reg <= load_value;
            overflow_reg <= 1'b0;
        end else if (enable) begin
            if (count_reg == MAX_COUNT) begin
                count_reg <= '0;
                overflow_reg <= 1'b1;
            end else begin
                count_reg <= count_reg + 1'b1;
                overflow_reg <= 1'b0;
            end
        end
    end

    // Combinational outputs
    assign count = count_reg;
    assign overflow = overflow_reg;

endmodule
