// Relative path design module
module relative_design (
    input logic clk,
    input logic reset,
    output logic [3:0] counter
);

    always_ff @(posedge clk) begin
        if (reset)
            counter <= 4'b0;
        else
            counter <= counter + 1;
    end

endmodule