// Testbench for VCS testing
module testbench;

    logic clk;
    logic reset;
    logic [7:0] data_in;
    logic [7:0] data_out;
    logic valid;

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // DUT instantiation
    core dut (
        .clk(clk),
        .reset(reset),
        .data_in(data_in),
        .data_out(data_out),
        .valid(valid)
    );

    // Test sequence
    initial begin
        reset = 1;
        data_in = 8'h00;
        #20 reset = 0;
        #10 data_in = 8'hFF;
        #10 data_in = 8'h55;
        #50 $finish;
    end

endmodule