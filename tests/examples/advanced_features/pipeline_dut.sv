// Pipeline DUT Module
// Multi-stage pipeline for testing complex sequential logic

module pipeline_dut #(
    parameter DATA_WIDTH = 32,
    parameter NUM_STAGES = 4
)(
    input  logic clk,
    input  logic reset,
    input  logic enable,
    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic valid_in,
    output logic [DATA_WIDTH-1:0] data_out,
    output logic valid_out,
    output logic [NUM_STAGES-1:0] stage_valid,
    output logic busy,
    output logic done
);

    // Pipeline stage data structure
    typedef struct packed {
        logic [DATA_WIDTH-1:0] data;
        logic valid;
        logic [3:0] stage_id;
    } pipeline_stage_t;

    // Pipeline registers
    pipeline_stage_t pipeline [0:NUM_STAGES-1];
    logic [DATA_WIDTH-1:0] processed_data;
    logic [3:0] stage_counter;
    logic processing;

    // Stage processing functions
    function automatic logic [DATA_WIDTH-1:0] stage0_process(input logic [DATA_WIDTH-1:0] data);
        return data + 8'h01;
    endfunction

    function automatic logic [DATA_WIDTH-1:0] stage1_process(input logic [DATA_WIDTH-1:0] data);
        return data << 1;
    endfunction

    function automatic logic [DATA_WIDTH-1:0] stage2_process(input logic [DATA_WIDTH-1:0] data);
        return data ^ 8'hAA;
    endfunction

    function automatic logic [DATA_WIDTH-1:0] stage3_process(input logic [DATA_WIDTH-1:0] data);
        return data + 8'h55;
    endfunction

    // Pipeline logic
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            // Reset all pipeline stages
            for (int i = 0; i < NUM_STAGES; i++) begin
                pipeline[i].data <= '0;
                pipeline[i].valid <= 1'b0;
                pipeline[i].stage_id <= '0;
            end
            processed_data <= '0;
            stage_counter <= '0;
            processing <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            valid_out <= 1'b0;
            data_out <= '0;
            stage_valid <= '0;
        end else if (enable) begin
            // Stage 0: Input processing
            if (valid_in) begin
                pipeline[0].data <= stage0_process(data_in);
                pipeline[0].valid <= 1'b1;
                pipeline[0].stage_id <= 4'd0;
                busy <= 1'b1;
            end else begin
                pipeline[0].valid <= 1'b0;
            end

            // Stage 1: Shift processing
            if (pipeline[0].valid) begin
                pipeline[1].data <= stage1_process(pipeline[0].data);
                pipeline[1].valid <= 1'b1;
                pipeline[1].stage_id <= 4'd1;
            end else begin
                pipeline[1].valid <= 1'b0;
            end

            // Stage 2: XOR processing
            if (pipeline[1].valid) begin
                pipeline[2].data <= stage2_process(pipeline[1].data);
                pipeline[2].valid <= 1'b1;
                pipeline[2].stage_id <= 4'd2;
            end else begin
                pipeline[2].valid <= 1'b0;
            end

            // Stage 3: Final processing
            if (pipeline[2].valid) begin
                pipeline[3].data <= stage3_process(pipeline[2].data);
                pipeline[3].valid <= 1'b1;
                pipeline[3].stage_id <= 4'd3;
            end else begin
                pipeline[3].valid <= 1'b0;
            end

            // Output stage
            if (pipeline[NUM_STAGES-1].valid) begin
                data_out <= pipeline[NUM_STAGES-1].data;
                valid_out <= 1'b1;
                processed_data <= pipeline[NUM_STAGES-1].data;
                stage_counter <= stage_counter + 1'b1;
                
                if (stage_counter >= 4'd10) begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end
            end else begin
                valid_out <= 1'b0;
            end

            // Update stage valid signals
            for (int i = 0; i < NUM_STAGES; i++) begin
                stage_valid[i] <= pipeline[i].valid;
            end
        end
    end

    // Combinational logic for monitoring
    always_comb begin
        processing = |stage_valid;
    end

endmodule

// Testbench for pipeline DUT
module pipeline_dut_tb;

    // Test parameters
    localparam DATA_WIDTH = 32;
    localparam NUM_STAGES = 4;

    // Clock and control signals
    logic clk;
    logic reset;
    logic enable;
    logic [DATA_WIDTH-1:0] data_in;
    logic valid_in;
    logic [DATA_WIDTH-1:0] data_out;
    logic valid_out;
    logic [NUM_STAGES-1:0] stage_valid;
    logic busy;
    logic done;

    // DUT instantiation
    pipeline_dut #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_STAGES(NUM_STAGES)
    ) dut (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .data_in(data_in),
        .valid_in(valid_in),
        .data_out(data_out),
        .valid_out(valid_out),
        .stage_valid(stage_valid),
        .busy(busy),
        .done(done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test stimulus
    initial begin
        // Initialize
        reset = 1;
        enable = 0;
        data_in = 0;
        valid_in = 0;

        #20;
        reset = 0;
        #10;

        // Enable the pipeline
        enable = 1;

        // Send test data through pipeline
        for (int i = 0; i < 15; i++) begin
            @(posedge clk);
            data_in = i * 10;
            valid_in = 1;

            @(posedge clk);
            valid_in = 0;

            // Wait a few cycles between inputs
            repeat(2) @(posedge clk);
        end

        // Wait for pipeline to complete
        wait(done);

        $display("Pipeline DUT test completed successfully");
        $finish;
    end

    // Monitor pipeline activity
    always @(posedge clk) begin
        if (valid_out) begin
            $display("Pipeline output: data=%h, valid=%b, stage_valid=%b", 
                     data_out, valid_out, stage_valid);
        end
    end

    // Monitor pipeline stages
    always @(posedge clk) begin
        if (|stage_valid) begin
            $display("Pipeline stages active: %b, busy=%b", stage_valid, busy);
        end
    end

endmodule
