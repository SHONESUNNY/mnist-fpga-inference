`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.10.2026 17:12:53
// Design Name: 
// Module Name: uart_rx
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module uart_rx #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer BAUD   = 115_200
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       rx,

    output logic [7:0] data_out,
    output logic       data_valid
);

    localparam integer CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam integer HALF_BIT     = CLKS_PER_BIT / 2;
    localparam integer COUNT_W      = $clog2(CLKS_PER_BIT);

    typedef enum logic [1:0] {
        IDLE,
        START_BIT,
        DATA_BITS,
        STOP_BIT
    } state_t;

    state_t state;

    logic rx_meta, rx_sync;
    logic [COUNT_W-1:0] clk_count;
    logic [2:0] bit_index;
    logic [7:0] rx_shift;

    // Synchronize asynchronous UART input to the FPGA clock.
    always_ff @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            clk_count  <= '0;
            bit_index  <= '0;
            rx_shift   <= '0;
            data_out   <= '0;
            data_valid <= 1'b0;
        end else begin
            // Default: data_valid is a one-clock pulse.
            data_valid <= 1'b0;

            case (state)

                IDLE: begin
                    clk_count <= '0;
                    bit_index <= '0;

                    // UART idle is logic HIGH.
                    if (!rx_sync)
                        state <= START_BIT;
                end

                START_BIT: begin
                    // Verify the start bit near its center.
                    if (clk_count == HALF_BIT - 1) begin
                        clk_count <= '0;

                        if (!rx_sync)
                            state <= DATA_BITS;
                        else
                            state <= IDLE;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                DATA_BITS: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        rx_shift[bit_index] <= rx_sync;

                        if (bit_index == 3'd7)
                            state <= STOP_BIT;
                        else
                            bit_index <= bit_index + 1'b1;
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                STOP_BIT: begin
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= '0;
                        state <= IDLE;

                        // Accept only a valid HIGH stop bit.
                        if (rx_sync) begin
                            data_out   <= rx_shift;
                            data_valid <= 1'b1;
                        end
                    end else begin
                        clk_count <= clk_count + 1'b1;
                    end
                end

                default: begin
                    state     <= IDLE;
                    clk_count <= '0;
                end

            endcase
        end
    end

endmodule
