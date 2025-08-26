// Template Engine Test Module
// Tests template engine caching and validation features

module template_module #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 4
)(
    input logic clk,
    input logic reset,
    input logic [DATA_WIDTH-1:0] data_in,
    output logic [DATA_WIDTH-1:0] data_out
);

    // Test parameter expressions (should use template-like generation)
    logic [DATA_WIDTH-1:0] data_reg;
    logic [ADDR_WIDTH-1:0] addr_counter;
    
    // Memory array to test template generation
    logic [DATA_WIDTH-1:0] memory [2**ADDR_WIDTH-1:0];
    
    // Sequential logic
    always_ff @(posedge clk) begin
        if (reset) begin
            data_reg <= '0;
            addr_counter <= '0;
            
            // Initialize memory
            for (int i = 0; i < 2**ADDR_WIDTH; i++) begin
                memory[i] <= '0;
            end
        end else begin
            memory[addr_counter] <= data_in;
            data_reg <= memory[addr_counter];
            addr_counter <= addr_counter + 1'b1;
        end
    end
    
    // Output
    assign data_out = data_reg;

endmodule
