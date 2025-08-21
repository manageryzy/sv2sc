module generate_adder #(
    parameter WIDTH = 4
)(
    input logic [WIDTH-1:0] a,
    input logic [WIDTH-1:0] b,
    input logic cin,
    output logic [WIDTH-1:0] sum,
    output logic cout
);

    logic [WIDTH:0] carry;
    assign carry[0] = cin;
    assign cout = carry[WIDTH];

    generate
        genvar i;
        for (i = 0; i < WIDTH; i++) begin : gen_fa
            full_adder fa_inst (
                .a(a[i]),
                .b(b[i]),
                .cin(carry[i]),
                .sum(sum[i]),
                .cout(carry[i+1])
            );
        end
    endgenerate

endmodule

module full_adder (
    input logic a,
    input logic b,
    input logic cin,
    output logic sum,
    output logic cout
);

    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (cin & (a ^ b));

endmodule