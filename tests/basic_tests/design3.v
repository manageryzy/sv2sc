// Design3 module for VCS testing (Verilog 2001)
module design3 (
    clk,
    reset,
    enable,
    count
);

    input clk;
    input reset; 
    input enable;
    output [7:0] count;
    
    reg [7:0] count;
    
    always @(posedge clk) begin
        if (reset)
            count <= 8'b0;
        else if (enable)
            count <= count + 1'b1;
    end

endmodule