module memory #(
    parameter WIDTH = 8,
    parameter DEPTH = 256
)(
    input logic clk,
    input logic reset,
    input logic write_enable,
    input logic read_enable,
    input logic [$clog2(DEPTH)-1:0] address,
    input logic [WIDTH-1:0] write_data,
    output logic [WIDTH-1:0] read_data
);

    logic [WIDTH-1:0] mem_array [DEPTH-1:0];
    logic [WIDTH-1:0] read_data_reg;

    // Write operation
    always_ff @(posedge clk) begin
        if (write_enable) begin
            mem_array[address] <= write_data;
        end
    end

    // Read operation
    always_ff @(posedge clk) begin
        if (reset) begin
            read_data_reg <= '0;
        end else if (read_enable) begin
            read_data_reg <= mem_array[address];
        end
    end

    assign read_data = read_data_reg;

    // Initialize memory (for simulation)
    initial begin
        for (int i = 0; i < DEPTH; i++) begin
            mem_array[i] = '0;
        end
    end

endmodule