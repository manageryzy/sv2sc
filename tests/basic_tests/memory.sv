// Memory module for VCS testing
module memory #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32
) (
    input  logic clk,
    input  logic reset,
    input  logic we,
    input  logic [ADDR_WIDTH-1:0] addr,
    input  logic [DATA_WIDTH-1:0] wdata,
    output logic [DATA_WIDTH-1:0] rdata
);

    logic [DATA_WIDTH-1:0] mem [2**ADDR_WIDTH-1:0];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < 2**ADDR_WIDTH; i++) begin
                mem[i] <= '0;
            end
        end else begin
            if (we)
                mem[addr] <= wdata;
            rdata <= mem[addr];
        end
    end

endmodule