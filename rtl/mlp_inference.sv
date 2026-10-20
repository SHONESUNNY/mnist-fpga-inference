`timescale 1ns/1ps

// Sequential INT8 MLP inference core for the exported 784 -> 64 -> 10 model.
//
// IMPORTANT: This RTL's requantization constants are taken from the current
// quantization_params.json file:
//   multiplier = 1715524261
//   right shift = 42
// If the quantized model is regenerated with different scales, update these.
//
// Pixel loading interface:
//   pulse pixel_we for each image byte, address 0..783 in row-major order.
//   After all 784 pixels have been written, pulse start for one clock.
//
// Reset is synchronous, active high.
module mlp_infer #(
    parameter W1_FILE = "weights_fc1.mem",
    parameter B1_FILE = "biases_fc1.mem",
    parameter W2_FILE = "weights_fc2.mem",
    parameter B2_FILE = "biases_fc2.mem"
) (
    input  logic clk,
    input  logic rst,
    input  logic start,

    input  logic pixel_we,
    input  logic [9:0] pixel_addr,
    input  logic [7:0] pixel_data,

    output logic busy,
    output logic done,
    output logic [3:0] predicted_digit,
    output logic signed [31:0] class_scores [0:9]
);
    localparam integer INPUTS = 784;
    localparam integer HIDDEN = 64;
    localparam integer CLASSES = 10;

    localparam logic signed [63:0] FC1_REQUANT_MULT = 64'sd1715524261;
    localparam integer FC1_REQUANT_SHIFT = 42;
    localparam logic [63:0] FC1_ROUNDING_BIAS = 64'd2199023255552; // 2^(42-1)

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_FC1_MAC,
        ST_FC1_ACCUM,
        ST_FC2_ACCUM,
        ST_FC1_REQUANT,
        ST_FC2_MAC,
        ST_FC2_STORE
    } state_t;

    state_t state;

    logic [7:0] image_mem [0:INPUTS-1];
    logic [7:0] hidden_mem [0:HIDDEN-1];

    logic signed [7:0]  weights_fc1 [0:(HIDDEN*INPUTS)-1];
    logic signed [31:0] biases_fc1  [0:HIDDEN-1];
    logic signed [7:0]  weights_fc2 [0:(CLASSES*HIDDEN)-1];
    logic signed [31:0] biases_fc2  [0:CLASSES-1];

    logic [6:0] neuron_idx; // 0..63 for FC1; 0..9 for FC2
    logic [9:0] mac_idx;    // 0..783 for FC1; 0..63 for FC2

    logic signed [31:0] accumulator;
    logic signed [31:0] best_score;
    logic [3:0] best_digit;

    integer fc1_weight_addr;
    integer fc2_weight_addr;
    logic signed [8:0]  pixel_signed;
    logic signed [7:0] weight_operand;
    logic signed [16:0] mac_product;
    logic signed [16:0] mac_product_reg;

    integer reset_i;

    initial begin
        $readmemh(W1_FILE, weights_fc1);
        $readmemh(B1_FILE, biases_fc1);
        $readmemh(W2_FILE, weights_fc2);
        $readmemh(B2_FILE, biases_fc2);
    end

    // Sign-extend each signed INT8 weight and unsigned UINT8 input to 32 bits
    // before multiplication. The product is bounded well inside signed INT32.
    always_comb begin
        fc1_weight_addr = 0;
        fc2_weight_addr = 0;
        pixel_signed = 9'd0;
        weight_operand = 8'sd0;
        mac_product = 17'sd0;

        case (state)
            ST_FC1_MAC: begin
                fc1_weight_addr = (neuron_idx * INPUTS) + mac_idx;
                pixel_signed = $signed({1'd0, image_mem[mac_idx]});
                weight_operand = weights_fc1[fc1_weight_addr];
                mac_product = pixel_signed * weight_operand;
            end

            ST_FC2_MAC: begin
                fc2_weight_addr = (neuron_idx * HIDDEN) + mac_idx;
                pixel_signed = $signed({1'b0, hidden_mem[mac_idx]});
                weight_operand = weights_fc2[fc2_weight_addr];
                mac_product = pixel_signed * weight_operand;
            end

            default: begin
                // Keep combinational outputs at zero outside MAC states.
            end
        endcase
    end

    // Reproduce the integer reference for FC1:
    //  1. ReLU on the signed accumulator.
    //  2. Multiply by M, round positive values to nearest (ties up/away from 0).
    //  3. Shift right 42 bits and saturate to UINT8 [0,255].
    // Because ReLU maps all non-positive accumulators to zero, negative
    // requantization does not need to be represented here.
    function automatic logic [7:0] requantize_hidden(
        input logic signed [31:0] acc_in
    );
        logic signed [63:0] product64;
        logic [63:0] rounded_value;
        begin
            if (acc_in <= 0) begin
                requantize_hidden = 8'd0;
            end else begin
                product64 = $signed(acc_in) * FC1_REQUANT_MULT;
                rounded_value = ($unsigned(product64) + FC1_ROUNDING_BIAS)>> FC1_REQUANT_SHIFT;
                if (rounded_value > 64'd255)
                    requantize_hidden = 8'd255;
                else
                    requantize_hidden = rounded_value[7:0];
            end
        end
    endfunction

    assign busy = (state != ST_IDLE);

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= ST_IDLE;
            neuron_idx <= 7'd0;
            mac_idx <= 10'd0;
            accumulator <= 32'sd0;
            best_score <= 32'sd0;
            best_digit <= 4'd0;
            predicted_digit <= 4'd0;
            mac_product_reg <= 17'sd0;
            
            done <= 1'b0;
            for (reset_i = 0; reset_i < CLASSES; reset_i = reset_i + 1)
                class_scores[reset_i] <= 32'sd0;
        end else begin
            done <= 1'b0;

            // The UART/top-level controller writes one pixel at a time.
            // Do not write while inference is active in the integrated design.
            if (pixel_we && (pixel_addr < INPUTS))
                image_mem[pixel_addr] <= pixel_data;

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        neuron_idx <= 7'd0;
                        mac_idx <= 10'd0;
                        accumulator <= biases_fc1[0];
                        best_score <= 32'sh80000000;
                        best_digit <= 4'd0;
                        state <= ST_FC1_MAC;
                    end
                end

                ST_FC1_MAC: begin
                mac_product_reg <= mac_product;
                state <= ST_FC1_ACCUM ;
                end

                ST_FC1_ACCUM: begin
                accumulator <= accumulator + $signed({{15{mac_product_reg[16]}}, mac_product_reg});
                 if (mac_idx == 10'd783) begin
        // All 784 inputs processed.
                    state <= ST_FC1_REQUANT;
                 end else begin
                     mac_idx <= mac_idx + 10'd1;
                      state <= ST_FC1_MAC;
                   end
         end
                ST_FC1_REQUANT: begin
                    hidden_mem[neuron_idx] <= requantize_hidden(accumulator);
                    if (neuron_idx == 7'd63) begin
                        neuron_idx <= 7'd0;
                        mac_idx <= 10'd0;
                        accumulator <= biases_fc2[0];
                        state <= ST_FC2_MAC;
                    end else begin
                        neuron_idx <= neuron_idx + 7'd1;
                        mac_idx <= 10'd0;
                        accumulator <= biases_fc1[neuron_idx + 7'd1];
                        state <= ST_FC1_MAC;
                    end
                end

                ST_FC2_MAC: begin
                    mac_product_reg <= mac_product;
                     state <= ST_FC2_ACCUM;
                end
                
                ST_FC2_ACCUM : begin 
                  accumulator <= accumulator + $signed({{15{mac_product_reg[16]}}, mac_product_reg});
                  if(mac_idx == 10'd63) begin
                     state <= ST_FC2_STORE;
                       end
                  else begin
                     mac_idx <= mac_idx + 10'd1 ;
                     state <= ST_FC2_MAC ; 
                      end
            end      
                ST_FC2_STORE: begin
                    class_scores[neuron_idx] <= accumulator;
                    if (neuron_idx == 7'd0) begin
                        best_score <= accumulator;
                        best_digit <= 4'd0;
                    end else if (accumulator > best_score) begin
                        best_score <= accumulator;
                        best_digit <= neuron_idx[3:0];
                    end
               
                    if (neuron_idx == 7'd9) begin
                        // Strict greater-than preserves the first class on ties,
                        // matching argmax's first-maximum behavior.
                        if (accumulator > best_score)
                            predicted_digit <= neuron_idx[3:0];
                        else
                            predicted_digit <= best_digit;
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        neuron_idx <= neuron_idx + 7'd1;
                        mac_idx <= 10'd0;
                        accumulator <= biases_fc2[neuron_idx + 7'd1];
                        state <= ST_FC2_MAC;
                    end
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end
endmodule
