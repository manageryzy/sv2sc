// Test case for sensitivity extraction from timing controls
module sensitivity_test (
    input  logic clk,
    input  logic reset,
    input  logic enable,
    input  logic [7:0] data_in,
    output logic [7:0] data_out,
    
    // Different clock domains
    input  logic clk2,
    input  logic clk_fast,
    
    // Different reset signals
    input  logic rst_n,
    input  logic resetn
);

    logic [7:0] reg1, reg2, reg3;
    logic [3:0] counter;
    
    // Test 1: Standard posedge clk with reset
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            reg1 <= 8'b0;
        end else begin
            reg1 <= data_in;
        end
    end
    
    // Test 2: Negedge clock
    always @(negedge clk2) begin
        reg2 <= reg1;
    end
    
    // Test 3: Multiple signals in sensitivity
    always @(posedge clk_fast or negedge rst_n) begin
        if (!rst_n) begin
            reg3 <= 8'b0;
        end else if (enable) begin
            reg3 <= reg2;
        end
    end
    
    // Test 4: always_ff with implicit posedge
    always_ff @(posedge clk) begin
        if (resetn == 1'b0) begin
            counter <= 4'b0;
        end else begin
            counter <= counter + 1'b1;
        end
    end
    
    // Test 5: Combinational with multiple inputs
    always @(data_in or enable or reg1) begin
        if (enable)
            data_out = data_in;
        else
            data_out = reg1;
    end
    
    // Test 6: Edge-sensitive without reset
    always @(posedge clk) begin
        if (enable) begin
            reg1 <= data_in;
        end
    end
    
endmodule
