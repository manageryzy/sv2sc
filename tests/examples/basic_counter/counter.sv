module counter (
    input logic clk,
    input logic reset,
    input logic enable,
    output logic [7:0] count
);

    logic [7:0] count_reg;

    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            count_reg <= 8'b0;
        end else if (enable) begin
            count_reg <= count_reg + 1'b1;
        end
    end

    assign count = count_reg;

endmodule