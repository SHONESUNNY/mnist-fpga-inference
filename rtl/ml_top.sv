`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.10.2026 12:20:13
// Design Name: 
// Module Name: ml_top
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

`timescale 1ns/1ps

// Top-level UART -> MNIST MLP -> UART integration for the Boolean Board.
// Protocol (first version):
//   PC sends exactly 784 raw pixel bytes, row-major, each in [0,255].
//   FPGA returns exactly one raw byte with the class value 0..9.
//   No header/checksum is used. PC must wait for the result before sending
//   the next image. Trained weights and biases remain fixed in FPGA memory.
module ml_demo_top (
    input  logic       clk,
    input  logic [3:0] btn,
    input  logic       UART_rxd,
    output logic       UART_txd
);
    localparam integer CLK_HZ = 50_000_000;
    localparam integer BAUD   = 115_200;
    localparam logic [9:0] LAST_PIXEL_ADDR = 10'd783;
    logic clk_50mhz;
    logic clk_locked;
    
    
    clk_wiz_0 u_clk_wiz (
    .clk_in1  (clk),
    .reset    (1'b0),
    .clk_out1 (clk_50mhz),
    .locked   (clk_locked)
);

    // Boolean Board button 0 is active-high reset.
    logic rst;
    assign rst = btn[0] | ~clk_locked ;

    // UART receiver signals.
    logic [7:0] rx_data;
    logic       rx_valid;

    // UART transmitter signals.
    logic [7:0] tx_data;
    logic       tx_start;
    logic       tx_busy;

    // Inference-core interface.
    logic       infer_start;
    logic       infer_pixel_we;
    logic [9:0] infer_pixel_addr;
    logic [7:0] infer_pixel_data;
    logic       infer_busy;
    logic       infer_done;
    logic [3:0] predicted_digit;
    logic signed [31:0] class_scores [0:9];

    // Address of the next pixel to store, 0 through 783.
    logic [9:0] image_count;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_RECEIVE_IMAGE,
        ST_WAIT_INFERENCE,
        ST_LAUNCH_TX,
        ST_WAIT_TX_BUSY,
        ST_WAIT_TX_DONE
    } state_t;
    state_t state;

    uart_rx #(
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD)
    ) receiver (
        .clk(clk_50mhz),
        .rst(rst),
        .rx(UART_rxd),
        .data_out(rx_data),
        .data_valid(rx_valid)
    );

    uart_tx #(
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD)
    ) transmitter (
        .clk(clk_50mhz),
        .rst(rst),
        .data_in(tx_data),
        .start(tx_start),
        .tx(UART_txd),
        .busy(tx_busy)
    );

    // Keep these filenames in the Vivado project / memory-file search path.
    mlp_infer #(
        .W1_FILE("weights_fc1.mem"),
        .B1_FILE("biases_fc1.mem"),
        .W2_FILE("weights_fc2.mem"),
        .B2_FILE("biases_fc2.mem")
    ) inference_core (
        .clk(clk_50mhz),
        .rst(rst),
        .start(infer_start),
        .pixel_we(infer_pixel_we),
        .pixel_addr(infer_pixel_addr),
        .pixel_data(infer_pixel_data),
        .busy(infer_busy),
        .done(infer_done),
        .predicted_digit(predicted_digit),
        .class_scores(class_scores)
    );

    // In IDLE, the first valid UART byte is pixel 0. In RECEIVE_IMAGE,
    // subsequent bytes fill pixels 1..783. The core writes the image memory
    // on the same edge that it observes infer_pixel_we.
    always_comb begin
        infer_pixel_we   = 1'b0;
        infer_pixel_addr = image_count;
        infer_pixel_data = rx_data;

        if (rx_valid && ((state == ST_IDLE) || (state == ST_RECEIVE_IMAGE)))
            infer_pixel_we = 1'b1;
    end

    always_ff @(posedge clk_50mhz) begin
        if (rst) begin
            state       <= ST_IDLE;
            image_count <= 10'd0;
            infer_start <= 1'b0;
            tx_data     <= 8'd0;
            tx_start    <= 1'b0;
        end else begin
            // Both are one-clock requests; the submodules sample the previous
            // registered value at the next active clock edge.
            infer_start <= 1'b0;
            tx_start    <= 1'b0;

            case (state)
                ST_IDLE: begin
                    image_count <= 10'd0;
                    if (rx_valid) begin
                        // First byte was written to pixel address zero above.
                        image_count <= 10'd1;
                        state <= ST_RECEIVE_IMAGE;
                    end
                end

                ST_RECEIVE_IMAGE: begin
                    if (rx_valid) begin
                        if (image_count == LAST_PIXEL_ADDR) begin
                            // This byte is pixel 783. The inference core writes
                            // it on this edge; start is seen by the core next edge.
                            image_count <= 10'd0;
                            infer_start <= 1'b1;
                            state <= ST_WAIT_INFERENCE;
                        end else begin
                            image_count <= image_count + 10'd1;
                        end
                    end
                end

                ST_WAIT_INFERENCE: begin
                    if (infer_done)
                        state <= ST_LAUNCH_TX;
                end

                ST_LAUNCH_TX: begin
                    // TX samples this registered request on the next clock.
                    tx_data  <= {4'b0000, predicted_digit};
                    tx_start <= 1'b1;
                    state    <= ST_WAIT_TX_BUSY;
                end

                ST_WAIT_TX_BUSY: begin
                    // Avoid mistaking the pre-start idle cycle for TX completion.
                    if (tx_busy)
                        state <= ST_WAIT_TX_DONE;
                end

                ST_WAIT_TX_DONE: begin
                    if (!tx_busy) begin
                        // Ready for the next 784-byte image.
                        image_count <= 10'd0;
                        state <= ST_IDLE;
                    end
                end

                default: begin
                    state       <= ST_IDLE;
                    image_count <= 10'd0;
                    infer_start <= 1'b0;
                    tx_start    <= 1'b0;
                end
            endcase
        end
    end
endmodule

