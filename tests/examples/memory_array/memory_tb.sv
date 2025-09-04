// Memory Testbench
module memory_tb;

    logic clk, reset;
    logic write_enable, read_enable;
    logic [7:0] address, write_data, read_data;
    logic ready, busy, valid, error;
    logic [1:0] error_code;

    memory dut (
        .clk(clk),
        .reset(reset),
        .write_enable(write_enable),
        .read_enable(read_enable),
        .address(address),
        .write_data(write_data),
        .read_data(read_data),
        .ready(ready),
        .busy(busy),
        .valid(valid),
        .error(error),
        .error_code(error_code)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        reset = 1;
        write_enable = 0;
        read_enable = 0;
        address = 0;
        write_data = 0;
        
        #20;
        reset = 0;
        #10;
        
        // Test basic operations
        write_enable = 1;
        address = 5;
        write_data = 8'hAA;
        #10;
        write_enable = 0;
        
        read_enable = 1;
        address = 5;
        #10;
        read_enable = 0;
        
        #50;
        $finish;
    end

endmodule


