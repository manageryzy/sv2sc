// FSM DUT Module
// Comprehensive Finite State Machine for testing state transitions

module fsm_dut #(
    parameter NUM_STATES = 4,
    parameter DATA_WIDTH = 8
)(
    input  logic clk,
    input  logic reset,
    input  logic enable,
    input  logic [1:0] input_signal,
    input  logic [DATA_WIDTH-1:0] data_in,
    output logic [1:0] current_state,
    output logic [1:0] next_state,
    output logic [DATA_WIDTH-1:0] data_out,
    output logic valid_out,
    output logic busy,
    output logic done
);

    // State definitions
    typedef enum logic [1:0] {
        IDLE     = 2'b00,
        PROCESS  = 2'b01,
        WAIT     = 2'b10,
        COMPLETE = 2'b11
    } state_t;

    // Internal signals
    state_t current_state_reg;
    state_t next_state_reg;
    logic [DATA_WIDTH-1:0] data_out_reg;
    logic valid_out_reg;
    logic busy_reg;
    logic done_reg;
    logic [DATA_WIDTH-1:0] temp_data;
    logic [3:0] counter;

    // State transition logic
    always_comb begin
        case (current_state_reg)
            IDLE: begin
                if (enable && input_signal == 2'b01) begin
                    next_state_reg = PROCESS;
                end else begin
                    next_state_reg = IDLE;
                end
            end
            
            PROCESS: begin
                if (counter >= 4'd3) begin
                    next_state_reg = WAIT;
                end else begin
                    next_state_reg = PROCESS;
                end
            end
            
            WAIT: begin
                if (input_signal == 2'b10) begin
                    next_state_reg = COMPLETE;
                end else if (input_signal == 2'b11) begin
                    next_state_reg = IDLE;
                end else begin
                    next_state_reg = WAIT;
                end
            end
            
            COMPLETE: begin
                next_state_reg = IDLE;
            end
            
            default: begin
                next_state_reg = IDLE;
            end
        endcase
    end

    // Sequential logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            current_state_reg <= IDLE;
            data_out_reg <= '0;
            valid_out_reg <= 1'b0;
            busy_reg <= 1'b0;
            done_reg <= 1'b0;
            temp_data <= '0;
            counter <= '0;
        end else begin
            current_state_reg <= next_state_reg;
            
            case (current_state_reg)
                IDLE: begin
                    valid_out_reg <= 1'b0;
                    busy_reg <= 1'b0;
                    done_reg <= 1'b0;
                    counter <= '0;
                    if (enable && input_signal == 2'b01) begin
                        temp_data <= data_in;
                        busy_reg <= 1'b1;
                    end
                end
                
                PROCESS: begin
                    valid_out_reg <= 1'b0;
                    busy_reg <= 1'b1;
                    done_reg <= 1'b0;
                    counter <= counter + 1'b1;
                    
                    // Process data
                    case (counter)
                        4'd0: temp_data <= temp_data + 8'h01;
                        4'd1: temp_data <= temp_data << 1;
                        4'd2: temp_data <= temp_data ^ 8'hAA;
                        4'd3: temp_data <= temp_data + 8'h55;
                        default: temp_data <= temp_data; // No change for other values
                    endcase
                end
                
                WAIT: begin
                    valid_out_reg <= 1'b0;
                    busy_reg <= 1'b1;
                    done_reg <= 1'b0;
                end
                
                COMPLETE: begin
                    data_out_reg <= temp_data;
                    valid_out_reg <= 1'b1;
                    busy_reg <= 1'b0;
                    done_reg <= 1'b1;
                end
            endcase
        end
    end

    // Output assignments
    assign current_state = current_state_reg;
    assign next_state = next_state_reg;
    assign data_out = data_out_reg;
    assign valid_out = valid_out_reg;
    assign busy = busy_reg;
    assign done = done_reg;

endmodule
