// ALU DUT Module
// Comprehensive Arithmetic Logic Unit for testing various operations

module alu_dut #(
    parameter WIDTH = 32
)(
    input  logic clk,
    input  logic reset,
    input  logic enable,
    input  logic [3:0] opcode,
    input  logic [WIDTH-1:0] operand_a,
    input  logic [WIDTH-1:0] operand_b,
    input  logic [4:0] shift_amount,
    output logic [WIDTH-1:0] result,
    output logic zero_flag,
    output logic carry_flag,
    output logic overflow_flag,
    output logic negative_flag,
    output logic ready
);

    // Operation codes
    localparam OP_ADD  = 4'b0000;
    localparam OP_SUB  = 4'b0001;
    localparam OP_AND  = 4'b0010;
    localparam OP_OR   = 4'b0011;
    localparam OP_XOR  = 4'b0100;
    localparam OP_NOT  = 4'b0101;
    localparam OP_SLL  = 4'b0110;
    localparam OP_SRL  = 4'b0111;
    localparam OP_SRA  = 4'b1000;
    localparam OP_MUL  = 4'b1001;
    localparam OP_DIV  = 4'b1010;
    localparam OP_MOD  = 4'b1011;
    localparam OP_EQ   = 4'b1100;
    localparam OP_LT   = 4'b1101;
    localparam OP_LTU  = 4'b1110;
    localparam OP_NOP  = 4'b1111;

    // Internal signals
    logic [WIDTH-1:0] result_reg;
    logic [WIDTH:0] add_result;
    logic [WIDTH:0] sub_result;
    logic [WIDTH*2-1:0] mul_result;
    logic [WIDTH-1:0] div_result;
    logic [WIDTH-1:0] mod_result;
    
    // Flag registers
    logic zero_flag_reg;
    logic carry_flag_reg;
    logic overflow_flag_reg;
    logic negative_flag_reg;
    logic ready_reg;

    // Sequential logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            result_reg <= '0;
            zero_flag_reg <= 1'b0;
            carry_flag_reg <= 1'b0;
            overflow_flag_reg <= 1'b0;
            negative_flag_reg <= 1'b0;
            ready_reg <= 1'b0;
        end else if (enable) begin
            case (opcode)
                OP_ADD: begin
                    add_result = operand_a + operand_b;
                    result_reg <= add_result[WIDTH-1:0];
                    carry_flag_reg <= add_result[WIDTH];
                    overflow_flag_reg <= (operand_a[WIDTH-1] == operand_b[WIDTH-1]) && 
                                       (add_result[WIDTH-1] != operand_a[WIDTH-1]);
                end
                
                OP_SUB: begin
                    sub_result = operand_a - operand_b;
                    result_reg <= sub_result[WIDTH-1:0];
                    carry_flag_reg <= sub_result[WIDTH];
                    overflow_flag_reg <= (operand_a[WIDTH-1] != operand_b[WIDTH-1]) && 
                                       (sub_result[WIDTH-1] == operand_b[WIDTH-1]);
                end
                
                OP_AND: begin
                    result_reg <= operand_a & operand_b;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_OR: begin
                    result_reg <= operand_a | operand_b;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_XOR: begin
                    result_reg <= operand_a ^ operand_b;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_NOT: begin
                    result_reg <= ~operand_a;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_SLL: begin
                    result_reg <= operand_a << shift_amount;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_SRL: begin
                    result_reg <= operand_a >> shift_amount;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_SRA: begin
                    result_reg <= $signed(operand_a) >>> shift_amount;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_MUL: begin
                    mul_result = operand_a * operand_b;
                    result_reg <= mul_result[WIDTH-1:0];
                    carry_flag_reg <= |mul_result[WIDTH*2-1:WIDTH];
                    overflow_flag_reg <= |mul_result[WIDTH*2-1:WIDTH];
                end
                
                OP_DIV: begin
                    if (operand_b != 0) begin
                        div_result = operand_a / operand_b;
                        result_reg <= div_result;
                        carry_flag_reg <= 1'b0;
                        overflow_flag_reg <= 1'b0;
                    end else begin
                        result_reg <= '1; // Division by zero
                        carry_flag_reg <= 1'b1;
                        overflow_flag_reg <= 1'b1;
                    end
                end
                
                OP_MOD: begin
                    if (operand_b != 0) begin
                        mod_result = operand_a % operand_b;
                        result_reg <= mod_result;
                        carry_flag_reg <= 1'b0;
                        overflow_flag_reg <= 1'b0;
                    end else begin
                        result_reg <= operand_a; // Modulo by zero
                        carry_flag_reg <= 1'b1;
                        overflow_flag_reg <= 1'b1;
                    end
                end
                
                OP_EQ: begin
                    result_reg <= (operand_a == operand_b) ? '1 : '0;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_LT: begin
                    result_reg <= ($signed(operand_a) < $signed(operand_b)) ? '1 : '0;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_LTU: begin
                    result_reg <= (operand_a < operand_b) ? '1 : '0;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
                
                OP_NOP: begin
                    result_reg <= result_reg; // No operation
                    carry_flag_reg <= carry_flag_reg;
                    overflow_flag_reg <= overflow_flag_reg;
                end
                
                default: begin
                    result_reg <= '0;
                    carry_flag_reg <= 1'b0;
                    overflow_flag_reg <= 1'b0;
                end
            endcase
            
            // Update flags
            zero_flag_reg <= (result_reg == 0);
            negative_flag_reg <= result_reg[WIDTH-1];
            ready_reg <= 1'b1;
        end else begin
            ready_reg <= 1'b0;
        end
    end

    // Output assignments
    assign result = result_reg;
    assign zero_flag = zero_flag_reg;
    assign carry_flag = carry_flag_reg;
    assign overflow_flag = overflow_flag_reg;
    assign negative_flag = negative_flag_reg;
    assign ready = ready_reg;

endmodule
