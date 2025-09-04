// Interface-based DUT Module
// Tests SystemVerilog interface translation to SystemC

// Data interface
interface data_if #(parameter WIDTH = 32);
    logic [WIDTH-1:0] data;
    logic valid;
    logic ready;
    
    modport master (
        output data, valid,
        input  ready
    );
    
    modport slave (
        input  data, valid,
        output ready
    );
endinterface

// Control interface
interface ctrl_if;
    logic reset;
    logic enable;
    logic busy;
    logic done;
    
    modport master (
        output reset, enable,
        input  busy, done
    );
    
    modport slave (
        input  reset, enable,
        output busy, done
    );
endinterface

// Main DUT module using interfaces
module interface_dut #(
    parameter DATA_WIDTH = 32,
    parameter BUFFER_SIZE = 8
)(
    input logic clk,
    data_if.master output_port,
    data_if.slave input_port,
    ctrl_if.slave control
);

    // Internal signals
    logic [DATA_WIDTH-1:0] buffer [0:BUFFER_SIZE-1];
    logic [$clog2(BUFFER_SIZE):0] write_ptr;
    logic [$clog2(BUFFER_SIZE):0] read_ptr;
    logic [DATA_WIDTH-1:0] processed_data;
    logic processing;
    logic [2:0] state;

    // State machine states
    localparam IDLE     = 3'b000;
    localparam READ     = 3'b001;
    localparam PROCESS  = 3'b010;
    localparam WRITE    = 3'b011;
    localparam WAIT     = 3'b100;

    // Sequential logic
    always_ff @(posedge clk or posedge control.reset) begin
        if (control.reset) begin
            write_ptr <= '0;
            read_ptr <= '0;
            processed_data <= '0;
            processing <= 1'b0;
            state <= IDLE;
            control.busy <= 1'b0;
            control.done <= 1'b0;
            output_port.valid <= 1'b0;
            input_port.ready <= 1'b0;
            
            // Initialize buffer
            for (int i = 0; i < BUFFER_SIZE; i++) begin
                buffer[i] <= '0;
            end
        end else begin
            case (state)
                IDLE: begin
                    control.busy <= 1'b0;
                    control.done <= 1'b0;
                    output_port.valid <= 1'b0;
                    input_port.ready <= 1'b1;
                    
                    if (control.enable) begin
                        state <= READ;
                        control.busy <= 1'b1;
                    end
                end
                
                READ: begin
                    if (input_port.valid && input_port.ready) begin
                        // Store data in buffer
                        buffer[write_ptr[$clog2(BUFFER_SIZE)-1:0]] <= input_port.data;
                        write_ptr <= write_ptr + 1'b1;
                        input_port.ready <= 1'b0;
                        state <= PROCESS;
                    end
                end
                
                PROCESS: begin
                    // Process the data
                    if (read_ptr != write_ptr) begin
                        processed_data <= buffer[read_ptr[$clog2(BUFFER_SIZE)-1:0]] + 8'h01;
                        read_ptr <= read_ptr + 1'b1;
                        state <= WRITE;
                    end else begin
                        state <= WAIT;
                    end
                end
                
                WRITE: begin
                    output_port.data <= processed_data;
                    output_port.valid <= 1'b1;
                    
                    if (output_port.ready) begin
                        output_port.valid <= 1'b0;
                        state <= READ;
                        input_port.ready <= 1'b1;
                    end
                end
                
                WAIT: begin
                    if (read_ptr == write_ptr) begin
                        state <= IDLE;
                        control.done <= 1'b1;
                    end else begin
                        state <= PROCESS;
                    end
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

// Testbench for interface DUT
module interface_dut_tb;

    // Clock and control signals
    logic clk;
    ctrl_if control();
    
    // Data interfaces
    data_if #(.WIDTH(32)) input_data();
    data_if #(.WIDTH(32)) output_data();
    
    // DUT instantiation
    interface_dut #(
        .DATA_WIDTH(32),
        .BUFFER_SIZE(8)
    ) dut (
        .clk(clk),
        .output_port(output_data.master),
        .input_port(input_data.slave),
        .control(control.slave)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize
        control.reset = 1;
        control.enable = 0;
        input_data.valid = 0;
        input_data.data = 0;
        output_data.ready = 1;
        
        #20;
        control.reset = 0;
        #10;
        
        // Enable the DUT
        control.enable = 1;
        
        // Send test data
        for (int i = 0; i < 10; i++) begin
            @(posedge clk);
            input_data.valid = 1;
            input_data.data = i * 10;
            
            @(posedge clk);
            input_data.valid = 0;
            
            // Wait for processing
            repeat(5) @(posedge clk);
        end
        
        // Wait for completion
        wait(control.done);
        
        $display("Interface DUT test completed successfully");
        $finish;
    end
    
    // Monitor
    always @(posedge clk) begin
        if (output_data.valid && output_data.ready) begin
            $display("Output: data=%h, valid=%b", output_data.data, output_data.valid);
        end
    end

endmodule
