module uart_tx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 115_200
)(
    input  logic       clk,
    input  logic       rst,
    input  logic [7:0] data_in,
    input  logic       start,

    output logic       tx,
    output logic       busy
);

    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam integer COUNT_W      = $clog2(CLKS_PER_BIT);

    typedef enum logic [1:0] {
        IDLE,
        START_BIT,
        DATA_BITS,
        STOP_BIT
    } state_t;

    state_t state;

    logic [COUNT_W-1:0] clk_count;
    logic [2:0] bit_index;
    logic [7:0] tx_shift;

    always_ff @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            clk_count <= '0;
            bit_index <= '0;
            tx_shift  <= '0;
            tx        <= 1'b1;
            busy      <= 1'b0;
        end else begin
            case (state)

                IDLE: begin
                    tx        <= 1'b1;
                    busy      <= 1'b0;
                    clk_count <= '0;

                    if (start) begin
                        tx_shift  <= data_in;
                        tx        <= 1'b0;  // Start bit
                        busy      <= 1'b1;
                        state     <= START_BIT;
                    end
                end

                START_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        bit_index <= 3'd0;
                        tx        <= tx_shift[0];
                        state     <= DATA_BITS;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                DATA_BITS: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;

                        if (bit_index == 3'd7) begin
                            tx    <= 1'b1;  // Stop bit
                            state <= STOP_BIT;
                        end else begin
                            bit_index <= bit_index + 1'b1;
                            tx <= tx_shift[bit_index + 1'b1];
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                STOP_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        tx        <= 1'b1;
                        busy      <= 1'b0;
                        state     <= IDLE;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                default: begin
                    state     <= IDLE;
                    tx        <= 1'b1;
                    busy      <= 1'b0;
                    clk_count <= '0;
                end

            endcase
        end
    end

endmodule