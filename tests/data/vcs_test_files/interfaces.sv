// Interface definitions for VCS testing
interface bus_if #(parameter WIDTH = 32);
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

// Module using interface
module bus_master #(parameter WIDTH = 32) (
    input logic clk,
    input logic reset,
    bus_if.master bus
);

    always_ff @(posedge clk) begin
        if (reset) begin
            bus.data <= '0;
            bus.valid <= 1'b0;
        end else if (bus.ready) begin
            bus.data <= $random;
            bus.valid <= 1'b1;
        end
    end

endmodule