// Enhanced Memory DUT Module
// Comprehensive memory module with burst operations, status flags, and error handling

module memory #(
    parameter WIDTH = 8,
    parameter DEPTH = 256,
    parameter BURST_LENGTH = 4
)(
    input  logic clk,
    input  logic reset,
    
    // Control signals
    input  logic write_enable,
    input  logic read_enable,
    input  logic burst_enable,
    input  logic [1:0] burst_type,  // 00: single, 01: increment, 10: decrement, 11: wrap
    
    // Address and data
    input  logic [$clog2(DEPTH)-1:0] address,
    input  logic [WIDTH-1:0] write_data,
    output logic [WIDTH-1:0] read_data,
    
    // Status and control
    output logic ready,
    output logic busy,
    output logic valid,
    output logic error,
    output logic [1:0] error_code,  // 00: none, 01: invalid_addr, 10: burst_overflow, 11: timeout
    
    // Burst control
    input  logic [BURST_LENGTH-1:0] burst_mask,
    output logic [BURST_LENGTH-1:0] burst_valid,
    output logic burst_done
);

    // Memory array
    logic [WIDTH-1:0] mem_array [DEPTH-1:0];
    logic [WIDTH-1:0] read_data_reg;
    
    // Control registers
    logic ready_reg;
    logic busy_reg;
    logic valid_reg;
    logic error_reg;
    logic [1:0] error_code_reg;
    logic [BURST_LENGTH-1:0] burst_valid_reg;
    logic burst_done_reg;
    
    // Burst control
    logic [$clog2(DEPTH)-1:0] burst_addr;
    logic [BURST_LENGTH-1:0] burst_counter;
    logic [2:0] burst_state;
    logic [7:0] timeout_counter;
    
    // Burst states
    localparam BURST_IDLE    = 3'b000;
    localparam BURST_READ    = 3'b001;
    localparam BURST_WRITE   = 3'b010;
    localparam BURST_WAIT    = 3'b011;
    localparam BURST_ERROR   = 3'b100;

    // Write operation
    always_ff @(posedge clk) begin
        if (write_enable && !busy_reg) begin
            if (address < DEPTH) begin
                mem_array[address] <= write_data;
                error_reg <= 1'b0;
                error_code_reg <= 2'b00;
            end else begin
                error_reg <= 1'b1;
                error_code_reg <= 2'b01; // Invalid address
            end
        end
    end

    // Read operation
    always_ff @(posedge clk) begin
        if (reset) begin
            read_data_reg <= '0;
            valid_reg <= 1'b0;
        end else if (read_enable && !busy_reg) begin
            if (address < DEPTH) begin
                read_data_reg <= mem_array[address];
                valid_reg <= 1'b1;
                error_reg <= 1'b0;
                error_code_reg <= 2'b00;
            end else begin
                error_reg <= 1'b1;
                error_code_reg <= 2'b01; // Invalid address
                valid_reg <= 1'b0;
            end
        end else begin
            valid_reg <= 1'b0;
        end
    end

    // Burst operation control
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            burst_state <= BURST_IDLE;
            burst_addr <= '0;
            burst_counter <= '0;
            burst_valid_reg <= '0;
            burst_done_reg <= 1'b0;
            timeout_counter <= '0;
            busy_reg <= 1'b0;
            ready_reg <= 1'b1;
        end else begin
            case (burst_state)
                BURST_IDLE: begin
                    if (burst_enable) begin
                        burst_state <= read_enable ? BURST_READ : BURST_WRITE;
                        burst_addr <= address;
                        burst_counter <= '0;
                        burst_valid_reg <= '0;
                        burst_done_reg <= 1'b0;
                        timeout_counter <= '0;
                        busy_reg <= 1'b1;
                        ready_reg <= 1'b0;
                    end else begin
                        ready_reg <= 1'b1;
                        busy_reg <= 1'b0;
                    end
                end
                
                BURST_READ: begin
                    if (burst_counter < BURST_LENGTH) begin
                        if (burst_addr < DEPTH) begin
                            read_data_reg <= mem_array[burst_addr];
                            burst_valid_reg[burst_counter] <= 1'b1;
                            burst_counter <= burst_counter + 1'b1;
                            
                            // Update address based on burst type
                            case (burst_type)
                                2'b01: burst_addr <= burst_addr + 1'b1; // Increment
                                2'b10: burst_addr <= burst_addr - 1'b1; // Decrement
                                2'b11: burst_addr <= (burst_addr + 1'b1) % DEPTH; // Wrap
                                default: burst_addr <= burst_addr; // Single
                            endcase
                        end else begin
                            burst_state <= BURST_ERROR;
                            error_reg <= 1'b1;
                            error_code_reg <= 2'b01; // Invalid address
                        end
                    end else begin
                        burst_state <= BURST_WAIT;
                        burst_done_reg <= 1'b1;
                    end
                end
                
                BURST_WRITE: begin
                    if (burst_counter < BURST_LENGTH) begin
                        if (burst_addr < DEPTH) begin
                            mem_array[burst_addr] <= write_data;
                            burst_valid_reg[burst_counter] <= 1'b1;
                            burst_counter <= burst_counter + 1'b1;
                            
                            // Update address based on burst type
                            case (burst_type)
                                2'b01: burst_addr <= burst_addr + 1'b1; // Increment
                                2'b10: burst_addr <= burst_addr - 1'b1; // Decrement
                                2'b11: burst_addr <= (burst_addr + 1'b1) % DEPTH; // Wrap
                                default: burst_addr <= burst_addr; // Single
                            endcase
                        end else begin
                            burst_state <= BURST_ERROR;
                            error_reg <= 1'b1;
                            error_code_reg <= 2'b01; // Invalid address
                        end
                    end else begin
                        burst_state <= BURST_WAIT;
                        burst_done_reg <= 1'b1;
                    end
                end
                
                BURST_WAIT: begin
                    timeout_counter <= timeout_counter + 1'b1;
                    if (timeout_counter >= 8'd255) begin
                        burst_state <= BURST_ERROR;
                        error_reg <= 1'b1;
                        error_code_reg <= 2'b11; // Timeout
                    end else if (!burst_enable) begin
                        burst_state <= BURST_IDLE;
                        busy_reg <= 1'b0;
                        ready_reg <= 1'b1;
                    end
                end
                
                BURST_ERROR: begin
                    burst_state <= BURST_IDLE;
                    busy_reg <= 1'b0;
                    ready_reg <= 1'b1;
                end
                
                default: begin
                    burst_state <= BURST_IDLE;
                    busy_reg <= 1'b0;
                    ready_reg <= 1'b1;
                end
            endcase
        end
    end

    // Output assignments
    assign read_data = read_data_reg;
    assign ready = ready_reg;
    assign busy = busy_reg;
    assign valid = valid_reg;
    assign error = error_reg;
    assign error_code = error_code_reg;
    assign burst_valid = burst_valid_reg;
    assign burst_done = burst_done_reg;

    // Initialize memory (for simulation)
    initial begin
        for (int i = 0; i < DEPTH; i++) begin
            mem_array[i] = '0;
        end
    end

endmodule