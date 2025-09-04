// Test case for NBA splitting - a complex module with multiple always_ff blocks
module complex_processor (
    input  logic clk,
    input  logic reset,
    
    // ALU interface
    input  logic [31:0] alu_a,
    input  logic [31:0] alu_b,
    input  logic [3:0]  alu_op,
    input  logic        alu_enable,
    output logic [31:0] alu_result,
    output logic        alu_done,
    
    // Memory interface
    input  logic        mem_read,
    input  logic        mem_write,
    input  logic [31:0] mem_addr,
    input  logic [31:0] mem_wdata,
    output logic [31:0] mem_rdata,
    output logic        mem_ready,
    
    // Control signals
    output logic [2:0]  state,
    output logic [31:0] pc,
    output logic [31:0] instruction
);

    // Internal registers
    logic [31:0] registers [0:31];
    logic [31:0] next_pc;
    logic [2:0]  next_state;
    logic [31:0] temp_result;
    
    // State machine states
    localparam IDLE   = 3'b000;
    localparam FETCH  = 3'b001;
    localparam DECODE = 3'b010;
    localparam EXEC   = 3'b011;
    localparam MEM    = 3'b100;
    localparam WB     = 3'b101;
    
    // ALU operations block - should be separate process
    always_ff @(posedge clk) begin
        if (reset) begin
            alu_result <= 32'b0;
            alu_done <= 1'b0;
            temp_result <= 32'b0;
        end else if (alu_enable) begin
            case (alu_op)
                4'b0000: temp_result <= alu_a + alu_b;  // ADD
                4'b0001: temp_result <= alu_a - alu_b;  // SUB
                4'b0010: temp_result <= alu_a & alu_b;  // AND
                4'b0011: temp_result <= alu_a | alu_b;  // OR
                4'b0100: temp_result <= alu_a ^ alu_b;  // XOR
                4'b0101: temp_result <= alu_a << alu_b[4:0];  // SLL
                4'b0110: temp_result <= alu_a >> alu_b[4:0];  // SRL
                4'b0111: temp_result <= $signed(alu_a) >>> alu_b[4:0];  // SRA
                default: temp_result <= 32'b0;
            endcase
            alu_result <= temp_result;
            alu_done <= 1'b1;
        end else begin
            alu_done <= 1'b0;
        end
    end
    
    // Program counter block - should be separate process
    always_ff @(posedge clk) begin
        if (reset) begin
            pc <= 32'b0;
            next_pc <= 32'b0;
        end else begin
            case (state)
                IDLE: next_pc <= pc;
                FETCH: next_pc <= pc + 4;
                DECODE: next_pc <= pc;
                EXEC: begin
                    if (instruction[6:0] == 7'b1100011) begin  // Branch
                        next_pc <= pc + {{20{instruction[31]}}, instruction[31:20]};
                    end else if (instruction[6:0] == 7'b1101111) begin  // JAL
                        next_pc <= pc + {{12{instruction[31]}}, instruction[31:12]};
                    end else begin
                        next_pc <= pc + 4;
                    end
                end
                default: next_pc <= pc;
            endcase
            pc <= next_pc;
        end
    end
    
    // State machine block - should be separate process
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            next_state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    next_state <= FETCH;
                end
                FETCH: begin
                    if (mem_ready) begin
                        next_state <= DECODE;
                    end else begin
                        next_state <= FETCH;
                    end
                end
                DECODE: begin
                    next_state <= EXEC;
                end
                EXEC: begin
                    if (instruction[6:0] == 7'b0000011 || instruction[6:0] == 7'b0100011) begin
                        next_state <= MEM;
                    end else begin
                        next_state <= WB;
                    end
                end
                MEM: begin
                    if (mem_ready) begin
                        next_state <= WB;
                    end else begin
                        next_state <= MEM;
                    end
                end
                WB: begin
                    next_state <= FETCH;
                end
                default: begin
                    next_state <= IDLE;
                end
            endcase
            state <= next_state;
        end
    end
    
    // Memory interface block - should be separate process
    always_ff @(posedge clk) begin
        if (reset) begin
            mem_rdata <= 32'b0;
            mem_ready <= 1'b0;
            instruction <= 32'b0;
        end else begin
            if (state == FETCH) begin
                if (mem_read) begin
                    instruction <= mem_wdata;  // Simulated memory read
                    mem_ready <= 1'b1;
                end else begin
                    mem_ready <= 1'b0;
                end
            end else if (state == MEM) begin
                if (mem_read) begin
                    mem_rdata <= mem_wdata;  // Simulated memory read
                    mem_ready <= 1'b1;
                end else if (mem_write) begin
                    // Memory write simulation
                    mem_ready <= 1'b1;
                end else begin
                    mem_ready <= 1'b0;
                end
            end else begin
                mem_ready <= 1'b0;
            end
        end
    end
    
    // Register file block - should be separate process
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < 32; i++) begin
                registers[i] <= 32'b0;
            end
        end else if (state == WB) begin
            // Write back to register file
            if (instruction[11:7] != 5'b0) begin  // Don't write to x0
                registers[instruction[11:7]] <= alu_result;
            end
        end
    end

endmodule
