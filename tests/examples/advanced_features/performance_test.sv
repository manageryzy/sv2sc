// Performance Test Module
// Tests performance profiling and measurement capabilities

module simple_counter #(
    parameter WIDTH = 8
)(
    input logic clk,
    input logic reset,
    input logic enable,
    output logic [WIDTH-1:0] count
);

    logic [WIDTH-1:0] counter_reg;
    
    // Simple sequential logic for performance testing
    always_ff @(posedge clk) begin
        if (reset) begin
            counter_reg <= '0;
        end else if (enable) begin
            counter_reg <= counter_reg + 1'b1;
        end
    end
    
    // Combinational output
    assign count = counter_reg;
    
    // Add some complexity for performance measurement
    logic [WIDTH-1:0] temp1, temp2, temp3;
    
    always_comb begin
        temp1 = counter_reg ^ {WIDTH{1'b1}};
        temp2 = temp1 + counter_reg;
        temp3 = temp2 & counter_reg;
    end

endmodule
