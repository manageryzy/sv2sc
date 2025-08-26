// Advanced SystemVerilog Constructs Test
// Tests detection and handling of advanced SystemVerilog features

// Package definition (should be detected and documented)
package test_pkg;
    parameter int PKG_WIDTH = 16;
    
    typedef struct packed {
        logic [7:0] data;
        logic valid;
    } data_packet_t;
    
    function automatic logic [7:0] increment(logic [7:0] value);
        return value + 1'b1;
    endfunction
endpackage

// Module using advanced constructs
module advanced_module 
    import test_pkg::*;
#(
    parameter WIDTH = PKG_WIDTH
)(
    input logic clk,
    input logic reset,
    input logic enable,
    input logic [7:0] data_in,
    output logic [7:0] data_out,
    output logic valid_out
);

    // Use package types
    data_packet_t packet_reg;
    logic [WIDTH-1:0] counter;
    
    // Sequential logic with function calls
    always_ff @(posedge clk) begin
        if (reset) begin
            packet_reg <= '0;
            counter <= '0;
        end else if (enable) begin
            packet_reg.data <= increment(data_in);
            packet_reg.valid <= 1'b1;
            counter <= counter + 1'b1;
        end else begin
            packet_reg.valid <= 1'b0;
        end
    end
    
    // Outputs
    assign data_out = packet_reg.data;
    assign valid_out = packet_reg.valid;
    
    // SystemVerilog assertions (commented out but should be detected)
    /*
    assert property (@(posedge clk) enable |-> ##1 valid_out);
    assert property (@(posedge clk) reset |=> !valid_out);
    
    // Immediate assertion
    always_comb begin
        assert (counter < 2**WIDTH) else $error("Counter overflow");
    end
    */
    
    // Coverage (should be detected)
    /*
    covergroup data_cg @(posedge clk);
        data_cp: coverpoint data_in;
        valid_cp: coverpoint valid_out;
    endgroup
    */

endmodule

// Simple class (should be detected)
/*
class test_transaction;
    rand logic [7:0] data;
    
    constraint data_c {
        data inside {[0:100]};
    }
    
    function new();
        data = 8'h00;
    endfunction
endclass
*/
