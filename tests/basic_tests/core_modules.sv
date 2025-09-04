// Core modules collection
module alu (
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [3:0]  op,
    output logic [31:0] result
);

    always_comb begin
        case (op)
            4'b0000: result = a + b;
            4'b0001: result = a - b;
            4'b0010: result = a & b;
            4'b0011: result = a | b;
            4'b0100: result = a ^ b;
            default: result = '0;
        endcase
    end

endmodule

module register_file (
    input  logic clk,
    input  logic we,
    input  logic [4:0] addr,
    input  logic [31:0] wdata,
    output logic [31:0] rdata
);

    logic [31:0] regs [31:0];

    always_ff @(posedge clk) begin
        if (we)
            regs[addr] <= wdata;
        rdata <= regs[addr];
    end

endmodule