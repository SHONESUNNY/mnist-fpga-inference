# mnist-fpga-inference
# Handwritten Digit Recognition on FPGA — Project Documentation

**Last updated:** 20 October 2026  
**Status:** End-to-end UART inference has worked on the physical board. Timing closure at the intended clock frequency still needs to be confirmed with a fresh post-implementation timing report.

---

## 1. Project summary

This project implements handwritten digit recognition using a small multilayer perceptron (MLP). The neural network is trained in Python/PyTorch, converted to a fixed-point integer representation, and implemented as sequential inference logic in SystemVerilog on a **Real Digital Boolean Board with an AMD/Xilinx Spartan-7 XC7S50 FPGA**.

A Python drawing application creates a 28 × 28 grayscale image. The PC sends the image's 784 pixel bytes to the FPGA over UART. The FPGA stores the pixels, executes the trained MLP using quantized weights and integer arithmetic, selects the class with the highest score, and returns one byte containing the predicted digit (0–9).



This confirms that the physical UART → image buffer → inference core → UART response path works for at least this test image. It does not, by itself, establish overall classification accuracy or prove that the design meets its timing constraints.

### Key project results so far

| Area | Current result |
|---|---|
| Model architecture | 784 → 64 → 10, hidden ReLU |
| Framework | PyTorch for training; custom integer reference for quantization checks |
| Floating-point test accuracy | 96.92% |
| Integer-reference test accuracy | 96.98% |
| Integer-vs-float prediction agreement | 99.90% |
| Quantization evaluation set | 10,000 MNIST test images |
| Exported RTL test vectors | 100 images |
| FPGA interface | UART, 115200 baud, 8-N-1 |
| Image input | 784 raw pixel bytes, row-major, unsigned 8-bit |
| Returned result | One raw byte in the range 0–9 |
| Live board test | Received 784 bytes and predicted digit 6 |
| Timing closure | **Not yet verified from a fresh final report** |

---

## 2. System overview

```text
             PC / Python
  ┌──────────────────────────────┐
  │ PyTorch training             │
  │ MNIST dataset                │
  │ Floating-point MLP           │
  └──────────────┬───────────────┘
                 │ trained checkpoint
                 ▼
  ┌──────────────────────────────┐
  │ Quantization / export        │
  │ INT8 weights                 │
  │ INT32 biases                 │
  │ Scaling and requantization   │
  │ .mem parameter files         │
  └──────────────┬───────────────┘
                 │
  ┌──────────────▼───────────────┐
  │ Python drawing application   │
  │ 28 × 28 image                │
  │ sent_image.bin: 784 bytes    │
  └──────────────┬───────────────┘
                 │ UART / USB serial
                 │ 115200 baud, 8-N-1
                 ▼
  ┌────────────────────────────────────────────────┐
  │ Boolean Board / Spartan-7 XC7S50               │
  │                                                │
  │ UART RX → image memory → sequential MLP core   │
  │                           │                    │
  │                           ▼                    │
  │                  digit class (0–9)             │
  │                           │                    │
  │ UART TX ◄─────────────────┘                    │
  └───────────────────────────┬────────────────────┘
                              │ one raw result byte
                              ▼
                     Python prints prediction
```

### What runs where?

- **Training runs on the PC.** PyTorch learns the model parameters from MNIST.
- **Quantization runs on the PC.** It converts floating-point parameters to integer values and checks the integer model against the floating-point model.
- **Inference runs on the FPGA.** The FPGA does not train the network. It consumes the fixed weights and biases stored in initialized memories and computes the class scores.
- **The PC application is the input/output front end.** It turns the drawing into bytes, sends them over UART, and reads the prediction byte.

---

## 3. Hardware and development environment

### Board and target

- **Board:** Real Digital Boolean Board
- **FPGA family/device:** Spartan-7 XC7S50
- **Board oscillator:** 100 MHz, top-level clock pin `clk`
- **UART baud rate:** 115200
- **Serial format:** 8 data bits, no parity, 1 stop bit (8-N-1)
- **PC serial port used in the shown session:** `COM4`
- **FPGA design tools:** AMD/Xilinx Vivado
- **Software:** Python, PyTorch, NumPy, and `pyserial`

The full part designation must match the exact FPGA on the board. A previous project note used `xc7s50csga324-1`; verify this against the selected part in the actual Vivado project rather than assuming the package/speed grade from the family name alone.

### Pin assignments used by the top-level design

The following constraints were used for the relevant top-level signals:

| Signal | Package pin | Purpose |
|---|---|---|
| `clk` | `F14` | 100 MHz board clock input |
| `btn[0]` | `J2` | Reset button |
| `UART_rxd` | `V12` | PC-to-FPGA UART receive input |
| `UART_txd` | `U11` | FPGA-to-PC UART transmit output |

Minimal XDC used during development:

```tcl
set_property -dict {PACKAGE_PIN F14 IOSTANDARD LVCMOS33} [get_ports {clk}]
set_property -dict {PACKAGE_PIN J2 IOSTANDARD LVCMOS33} [get_ports {btn[0]}]
set_property -dict {PACKAGE_PIN V12 IOSTANDARD LVCMOS33} [get_ports {UART_rxd}]
set_property -dict {PACKAGE_PIN U11 IOSTANDARD LVCMOS33} [get_ports {UART_txd}]

# The external board oscillator remains 100 MHz.
create_clock -name sys_clk_pin -period 10.000 -waveform {0 5} [get_ports {clk}]
```

**Constraint note:** Keep one primary clock definition on the actual board clock input. If the Clocking Wizard generates an internal clock, its generated-clock constraints should come from the IP configuration. Do not add another primary `create_clock` definition to a signal downstream of the Clocking Wizard. Also check all active XDC files for duplicate clock declarations.

---

## 4. Neural-network model

### Architecture

The trained network is a fully connected MLP:

```text
Input image:       28 × 28 = 784 values
        │
        ▼
Fully connected 1: 784 → 64
        │
        ▼
ReLU activation
        │
        ▼
Fully connected 2: 64 → 10
        │
        ▼
10 class scores
        │
        ▼
argmax(scores)
        │
        ▼
Predicted digit: 0–9
```

- **Input features:** the image's 784 pixels, flattened in row-major order.
- **Hidden layer:** 64 neurons with ReLU activation.
- **Output layer:** 10 class scores, one for each digit.
- **Prediction:** the class index with the greatest output score.

The core evaluates the network sequentially using a multiply-accumulate datapath rather than computing every neuron in one huge combinational expression.

### Operation count

- First fully connected layer: `784 × 64 = 50,176` weight multiplications.
- Second fully connected layer: `64 × 10 = 640` weight multiplications.
- Total weight multiplications per inference: **50,816**.

The current state-machine approach processes one product and one accumulation operation across separate states. Consequently, a product requires more than one clock cycle to complete its contribution to the running sum; this was a deliberate timing-oriented change discussed later.

---

## 5. Training and test results

The checkpoint reported by the quantization configuration is:

```text
image_processing/mnist_artifacts/mnist_mlp_best.pth
```

The metadata records checkpoint epoch 9. The saved model was evaluated on the 10,000-image MNIST test set.

| Evaluation | Reported result |
|---|---:|
| Floating-point model accuracy | 96.92% |
| Integer-reference accuracy | 96.98% |
| Integer/reference prediction agreement with float | 99.90% |
| Images in test set | 10,000 |
| Images exported for RTL checks | 100 |

These metrics come from the Python-side evaluation/quantization flow. They are **not yet an end-to-end FPGA accuracy measurement**. That requires sending a labelled test subset through the physical UART interface and comparing every received FPGA prediction against the ground-truth labels.

### Train/test hygiene

The intended workflow is to train on the training dataset, choose the checkpoint using the project's validation/checkpoint logic, use calibration data for quantization parameters, and reserve the MNIST test set for final evaluation. Do not tune quantization parameters repeatedly against the test labels and then continue reporting the same test accuracy as an unbiased final result.

---

## 6. Integer quantization and exported parameters

Quantization is necessary because the FPGA implementation uses fixed-width integer arithmetic instead of floating-point tensors.

### Input representation

The drawing application produces 28 × 28 pixel values stored as unsigned bytes:

- Data type: `uint8`
- Shape: `[28, 28]` before flattening
- Pixel order: row-major
- Range: 0 through 255
- Scale: `1 / 255 = 0.00392156862745098`
- Zero point: 0
- Pixel convention in the quantization metadata: black = 0, bright = 255
- Real-valued input interpretation: `pixel_uint8 / 255.0`

The byte order matters. A correct set of 784 pixel values in the wrong order describes a different image.

### Weight and bias formats

| Parameter | Shape | Stored type | Scale / interpretation |
|---|---|---|---|
| FC1 weights | `[64, 784]` | signed INT8 | `0.005920778109332708` |
| FC1 biases | `[64]` | signed INT32 | accumulator scale `2.3218737683657676e-05` |
| FC1 activation after ReLU and requantization | `[64]` | unsigned UINT8 | `0.05952529533236634`, zero point 0 |
| FC2 weights | `[10, 64]` | signed INT8 | `0.006981253154634491` |
| FC2 biases | `[10]` | signed INT32 | `0.00041556115581963226` |
| FC2 class scores | `[10]` | signed INT32 | same positive scale for all classes, so argmax can compare raw integer scores |

Weight quantization is symmetric per-tensor INT8 with zero point 0 and quantized range `[-127, 127]`. The configuration records NumPy round-to-nearest-even for offline weight/bias conversion. The runtime requantization logic instead uses round-to-nearest with ties away from zero.

### Why FC1 multiplies and shifts instead of dividing by 255

The FC1 accumulator is the sum of integer products plus an integer bias. Its numeric scale is not simply the scale of the input pixels; it depends on both the input scale and the weight scale. The result must then be converted into the representation expected by the next layer.

The project's fixed-point implementation represents that conversion as a precomputed integer multiplier followed by a right shift and saturation. This avoids a general-purpose divider in hardware and reproduces the calibrated scale conversion. A generic `/ 255` would not implement the actual input-scale × weight-scale to activation-scale conversion and would generally produce the wrong magnitude.

### FC1 requantization constants

The current exported constants are:

```text
Real multiplier:       0.00039006505644387284
Integer multiplier:    1715524261
Exponent:              -11
Right shift:           42
Rounding:              ties away from zero at runtime
Output clamp:          [0, 255]
```

The RTL uses a wide signed product for the accumulator-to-multiplier operation, adds the rounding term appropriate to the sign/rounding method, shifts right by 42 bits, then saturates the result to an unsigned 8-bit value. ReLU is also applied to the first-layer result.

**Do not treat these constants as universal.** If the network is retrained or a scale changes, regenerate the quantization files and update the constants used by the RTL together.

### Why FC2 does not requantize before argmax

The final layer produces 10 signed integer scores. All 10 scores use the same positive scale. Multiplying every class score by the same positive scale does not change their ordering, so the hardware can find the maximum directly from integer scores without converting them back to floating point.

### Parameter-memory files

The following exported files are used by the inference RTL:

```text
weights_fc1.mem
biases_fc1.mem
weights_fc2.mem
biases_fc2.mem
```

Their layouts are:

- `weights_fc1.mem`: 64 output neurons, each followed by 784 consecutive input weights; row-major order from PyTorch `[out, in]` weights.
- `biases_fc1.mem`: 64 signed 32-bit two's-complement words.
- `weights_fc2.mem`: 10 output neurons, each followed by 64 consecutive input weights; row-major order from PyTorch `[out, in]` weights.
- `biases_fc2.mem`: 10 signed 32-bit two's-complement words.

The RTL initializes these memories with `$readmemh`. Vivado must be able to locate the files in the relevant simulation/synthesis/elaboration working context. A synthesis log that reports all four memories were read is a useful check, but does not prove their contents are ordered correctly; compare FPGA output against the integer reference as a functional check.

---

## 7. Python drawing app and UART protocol

### Drawing application and image file

The Python drawing application (`digit_draw_app.py` in the current workflow; some older script help text calls it `draw_app.py`) captures a drawn digit and converts it to a 28 × 28 image. The raw output is saved as:

```text
sent_image.bin
```

The file contains exactly **784 bytes**, with one unsigned byte per pixel in row-major order. It is a raw binary file, not a PNG/JPEG file and not a text file of decimal values.

The drawing application's preprocessing needs to remain consistent with the MNIST preprocessing used during training. In particular, image size, centering, polarity, intensity range, and flattening order all affect prediction quality.

### UART packet format

The current first-version protocol is intentionally simple:

1. PC opens the serial port.
2. PC sends exactly 784 raw pixel bytes.
3. FPGA receives and stores those bytes in image memory.
4. FPGA starts inference after the last byte is stored.
5. FPGA sends one raw byte containing the predicted digit, 0–9.
6. PC reads one byte and prints the integer value.

There is currently no start marker, packet length field, checksum, sequence number, or explicit acknowledgement. The sender assumes the receiver is ready for one image and that no old response byte remains queued. A future more robust protocol should add framing and an explicit ready/busy or acknowledgement mechanism if repeated or streaming inference is required.

### Python sender

The current sender script is `send_digit_uart.py`. Typical invocation from the appropriate project directory:

```powershell
python send_digit_uart.py --port COM4 --baud 115200 --image digit_output/sent_image.bin
```

The script checks that the input file exists and contains exactly 784 bytes, opens the serial port using the selected baud rate, writes the payload, flushes the serial output, then reads one response byte. It rejects a missing response and rejects result bytes above 9.

The successful live output was:

```text
Sent 784 image bytes
FPGA predicted digit: 6
```

### Current transfer scale

At 115200 baud using 8-N-1 framing, each UART data byte consumes 10 serial bits (start bit + 8 data bits + stop bit). Transmitting 784 bytes therefore has a theoretical wire time of:

```text
784 × 10 / 115200 ≈ 0.0681 seconds ≈ 68 ms
```

This excludes USB/driver buffering, Python overhead, inter-byte gaps, and FPGA processing. The image transfer itself can therefore be longer than a small sequential MLP's compute time.

---

8. RTL organization and inference flow
The current design implements a sequential, quantized 784–64–10 MLP on the Spartan-7 FPGA. The principal RTL/IP blocks are:
- uart_rx.sv — receives serial data and pulses a valid-byte signal when a byte is assembled.
- uart_tx.sv — serializes the predicted digit and transmits it to the PC.
- mlp_infer.sv — stores the 784 image pixels and performs quantized sequential MLP inference.
- ml_demo_top.sv — connects UART reception/transmission, image loading, inference control, reset, and board clocking.
- clk_wiz_0 — Clocking Wizard IP, if enabled in the active Vivado project configuration.
The exact Vivado project can include additional constraints, generated IP files, and source-file locations. The list above describes the main logical blocks.
Top-level controller
At a high level, the top-level controller:
1. Waits for a received UART byte.
2. Writes each byte into the inference core's image memory using a pixel address.
3. Increments the image counter after each byte.
4. After byte 784 is received (image_count == 783 in a zero-based counter), issues the one-cycle inference start pulse.
5. Waits for the MLP core to assert done, captures the predicted digit, and starts UART transmission.
6. Waits for UART transmission to finish before returning to idle.
The MLP core must not be started until all 784 pixel values have been loaded. The key interface signals include pixel write enable/address/data, inference start, busy/done, and predicted digit.
Inference state machine
The timing-optimized inference FSM uses separate states for multiplication, accumulation, and requantization work. The current state list includes:
- ST_IDLE
- ST_FC1_MAC
- ST_FC1_ACCUM
- ST_FC1_REQUANT
- ST_FC1_REQUANT_FINISH
- ST_FC2_MAC
- ST_FC2_ACCUM
- ST_FC2_STORE
This is eight states, which fits in a 3-bit state encoding. The FC1 neuron processes 784 input/weight products; each FC2 output neuron processes 64 products. Each accumulator starts with the corresponding INT32 bias. FC1 results are requantized, passed through ReLU, and stored as unsigned 8-bit hidden activations. FC2 scores are compared to determine the predicted class.

FC1 requantization was pipelined to shorten the critical combinational path: ST_FC1_REQUANT registers the accumulator-times-multiplier product, and ST_FC1_REQUANT_FINISH performs rounding, shifting, saturation, and hidden-activation storage. This adds a cycle but separates the wide multiply from the remaining requantization logic.
Signedness and datapath widths
The timing-optimized datapath avoids using signed 32-bit operands for the activation/weight multiplication.
Signal/data	Width	Interpretation
image_mem[...]	8 bits	Unsigned pixel, 0–255
hidden_mem[...]	8 bits	Unsigned quantized activation, 0–255
pixel_signed	9 bits	Positive signed representation of an unsigned activation, formed by prepending 1'b0
Weight operand	8 bits	Signed INT8 weight
Multiplication result	17 bits	Signed product of the 9-bit activation representation and signed 8-bit weight
Accumulator	32 bits	Signed bias plus accumulated products
FC1/FC2 biases	32 bits	Signed INT32 bias values


An unsigned 8-bit activation can reach 255, which cannot be represented as a positive signed 8-bit value. Prepending a zero creates a signed 9-bit value that preserves the full range. The signed 17-bit product is sign-extended before being added to the 32-bit accumulator:
$signed({{15{mac_product_reg[16]}}, mac_product_reg})
The product register separates multiplication from accumulation across FSM states. Correct signed extension is essential because the INT8 weight can be negative.
FPGA resource mapping
The latest reported implementation used block RAM for weight storage and DSP resources for arithmetic, rather than relying entirely on LUT-based weight storage. The reported placed-design utilization was approximately:
Resource	Reported utilization
LUTs	414
Flip-flops	216
BRAM tiles	17
DSPs	4


These figures describe the reported implementation run and should be regenerated from the final routed run if exact final resource numbers are needed.
9. Timing-debug history
Timing was the main implementation challenge. Earlier versions failed the 100 MHz setup constraint, despite functional progress. The design was optimized by narrowing the datapath, registering the MAC product, separating accumulation into another FSM state, and pipelining FC1 requantization.
Timing report history
Stage / report	WNS	TNS	Failing setup endpoints	Clock summary	Interpretation
Initial implementation	−8.474 ns	−748.921 ns	104	sys_clk_pin, 10 ns / 100 MHz	Significant setup violations; timing failed.
After datapath-width reduction and split MAC/accumulate (intermediate)	About −4.263 ns	Not reliably preserved in the notes	25 reported	100 MHz constraint	Improved worst slack, but timing still failed.
Later report during Clocking Wizard investigation	−5.986 ns	−162.863 ns	32	sys_clk_pin, 10 ns / 100 MHz	Still failed timing; this report did not prove that the MLP was timed by a 50 MHz generated clock.
Latest placed timing summary	+1.619 ns	0 ns	0 of 1,075 setup endpoints	sys_clk_pin, 10 ns / 100 MHz	All user-specified setup constraints were met in the placed timing report. Post-route timing still needs verification.


The WNS improved by 10.093 ns from the initial −8.474 ns result to the latest placed result of +1.619 ns. These reports were generated at different implementation stages/revisions, so the trend records overall progress rather than isolating the effect of one RTL change.
9.1 Initial failure: −8.474 ns WNS
The first documented timing summary showed:
WNS(ns)                    -8.474
TNS(ns)                  -748.921
TNS Failing Endpoints          104
TNS Total Endpoints            1717
WHS(ns)                    0.056
THS(ns)                    0.000
Clock period                10.000 ns
Frequency                 100.000 MHz
Timing constraints are not met.
The critical path passed through the multiplier/DSP path toward the accumulator, with additional LUT and MUXF7/MUXF8 logic. One reported path delay was approximately 18.256 ns against a 10 ns clock period. A high-fanout net was also reported with roughly 6,272 loads, contributing to routing and capacitance pressure.

The worst setup path exceeded the allowed clock period, so the placed implementation could not be considered timing-clean at 100 MHz under those constraints.
9.2 Datapath-width reduction
The early RTL represented the pixel operand, weight operand, and product using unnecessarily wide signed 32-bit values. The datapath was narrowed to match the actual numeric ranges:
- Image and hidden activations remain unsigned 8-bit values in memory.
- The activation operand is converted to signed 9-bit by prepending 1'b0.
- The weight operand is signed 8-bit.
- The product is signed 17-bit.
- The signed product is explicitly extended to 32 bits before accumulation.
This change reduced the width of the multiplier inputs and made signed arithmetic explicit. The next intermediate report improved to approximately −4.263 ns WNS, but still had setup violations.
9.3 Separate multiplication and accumulation
The design registers the multiplication result and adds that registered result to the accumulator in a subsequent FSM state. The FC1 sequence is conceptually:
ST_FC1_MAC: begin
    mac_product_reg <= mac_product;
    state <= ST_FC1_ACCUM;
end

ST_FC1_ACCUM: begin
    accumulator <= accumulator +
        $signed({{15{mac_product_reg[16]}}, mac_product_reg});

    if (mac_idx == 10'd783)
        state <= ST_FC1_REQUANT;
    else begin
        mac_idx <= mac_idx + 10'd1;
        state <= ST_FC1_MAC;
    end
end
FC2 follows the same approach, with the last input index at 63. This uses additional clock cycles per neuron, but keeps the multiplication and accumulator addition in separate sequential stages. The intermediate slack improved, although later reports showed the design still needed additional optimization.
9.4 FC1 requantization pipeline
The FC1 requantization path was another critical path. The computation includes multiplying the signed accumulator by the requantization multiplier, adding the rounding term, shifting, and saturating the result to the 8-bit activation range.

The multiply was separated from the rounding/shift/saturation logic using a 64-bit register, requant_product_reg:
- ST_FC1_REQUANT computes and registers the accumulator-times-multiplier product. Non-positive accumulators are handled as zero for the ReLU path.
- ST_FC1_REQUANT_FINISH applies the defined rounding and right shift, saturates to the unsigned 8-bit range, stores the hidden activation, and advances to the next neuron or FC2.
This pipeline shortened the FC1 requantization combinational path. After this change, the latest reported placed timing summary showed positive WNS and zero TNS at the 100 MHz constraint.
9.5 Latest placed timing results
The latest available report, ml_demo_top_timing_summary_placed.rpt, reports:
Clock:                       sys_clk_pin
Clock period:                10.000 ns
Clock frequency:             100.000 MHz
WNS:                         +1.619 ns
TNS:                          0.000 ns
Failing setup endpoints:      0 of 1,075
WHS:                         +0.187 ns
THS:                          0.000 ns
WPWS:                        +3.870 ns
TPWS:                         0.000 ns
The report states that all user-specified timing constraints are met for this placed implementation. The latest critical path was from the BRAM/weight-address read path (RAMB36E1) to mac_product_reg_reg[16]/D. The reported path delay was 8.392 ns, comprising approximately 3.573 ns logic delay and 4.819 ns routing delay across 11 logic levels. The path included BRAM, CARRY4, and LUT elements.

The remaining path is therefore no longer the earlier multiplier-to-accumulator path; it is associated with the weight-memory read/address logic feeding the MAC product register.

Important scope note: this is a placed timing report, not a post-route timing summary. It is evidence that the placement-stage timing analysis meets the specified 100 MHz setup constraint, but final timing closure should be confirmed after routing using a fresh post-route timing report from the final implementation run.
9.6 Clock constraints and Clocking Wizard checks
The board input clock is constrained with a 10 ns period:
create_clock -name sys_clk_pin -period 10.000 \
    -waveform {0 5} [get_ports {clk}]
The latest timing summary recognizes sys_clk_pin at 100 MHz and reports no generated clocks. This is consistent with a single-clock 100 MHz timing analysis. If the active design still expects clk_wiz_0 to drive part of the design at 50 MHz, the generated 20 ns clock must appear in report_clocks and the inference/UART registers must be analyzed against the intended clock. Do not claim 50 MHz operation based only on a Clocking Wizard configuration; verify the active timing report.

A previous Vivado warning indicated a possible duplicate/downstream clock definition, similar to:
Invalid clock redefinition on a clock tree. The primary clock
u_clk_wiz/inst/clk_in1 is defined downstream of clock sys_clk_pin
and overrides its insertion delay and/or waveform definition.
The root primary clock should be constrained at the top-level clk input. The Clocking Wizard's generated constraints should describe any output clock. Check active XDC files for duplicate create_clock commands and confirm that the generated IP constraints are enabled.

A previous top-level revision also used clk for the controller while the UART and inference blocks used clk_50mhz. That mixed-clock arrangement should not be retained unless an intentional clock-domain crossing design is present. If the system is intended to be synchronous, clock the controller and submodules from the same clock and ensure UART divider parameters match that clock frequency.
9.7 Current status and remaining verification
Item	Current evidence/status
Datapath widths and signed extension	Narrowed activation/weight/product widths are part of the timing-optimized RTL revision.
Multiplication versus accumulation	Product register separates the MAC product from accumulation.
FC1 requantization path	Pipelined across ST_FC1_REQUANT and ST_FC1_REQUANT_FINISH.
Timing at placement stage	Latest report: WNS +1.619 ns, TNS 0, zero failing setup endpoints at the 100 MHz constraint.
Post-route timing closure	Not yet confirmed from the evidence recorded here. Generate and inspect the final post-route timing summary.
Clocking Wizard / generated clock	Latest report lists only the 100 MHz sys_clk_pin clock. Verify a 50 MHz generated clock if the current design is intended to use one.
Physical UART-to-inference path	A live UART test returned predicted digit 6 for at least one image. This confirms one end-to-end test, not full model validation.
Functional equivalence to integer reference	Still requires batch testing across the generated test vectors and comparison of FPGA predictions with the software integer-reference model.
Unconstrained I/O	The timing checks reported one input port without an input delay and one output port without an output delay. Review the UART/reset interface constraints if full I/O timing coverage is required.


9.8 Commands for final verification
After the final implementation run, generate fresh reports from the active routed design in the Vivado Tcl Console:
report_clocks
report_clock_networks
report_timing_summary -delay_type max
report_utilization
check_timing
Before declaring final timing closure, verify that:
- the report comes from the final routed design, not an earlier placement run;
- the expected primary and generated clocks are present;
- WNS is non-negative and TNS is zero for the intended clock constraints;
- no clock-redefinition warning remains;
- the UART/controller/inference clocking arrangement is consistent;
- unconstrained input/output ports have been reviewed;
- multiple physical FPGA predictions match the integer-reference outputs.
Retain the post-route clock summary, worst setup path, utilization report, and any timing warnings in the repository. Functional operation and static timing closure are different forms of evidence: a successful UART inference test does not, by itself, prove timing closure or full numerical equivalence.
## 10. Earlier UART transport experiments

Before switching to the image-recognition payload, UART echo testing was used to validate the PC-to-FPGA-to-PC serial path.

The earlier echo design received bytes and echoed them back. Chunked transfers with 512-byte chunks were substantially more reliable than one unpaced large continuous transfer. One project note records a 32,768-byte transfer verified in about 3.39 seconds, approximately 9.43 KiB/s (another observed rate was about 9.46 KiB/s).

A separate continuous-transfer stress run showed corruption/incomplete reception:

```text
Bytes sent:       32768
Bytes received:   32726
Write/flush time: 2.485 s
Total time:       10.502 s
Received rate:    3.04 KiB/s
Mismatches:       23577
First mismatch at byte 9049
Missing bytes:    42
FAIL: Transfer incomplete or corrupted.
```

That test illustrated that sending faster or in larger unpaced bursts can expose buffering/throughput problems. The exact cause depended on the transmitter/receiver design and host/USB/serial buffering; it should not be attributed to a single cause without additional instrumentation.

The current digit protocol transfers only 784 bytes and has now succeeded end to end. The earlier echo result is useful background, but it does not guarantee that repeated image transactions, back-to-back inference, or sustained streaming are robust. Stress-test the actual inference protocol separately if the application needs repeated predictions.

---

## 11. Project files and their roles

The following names have appeared in the current workflow. Folder paths can vary by the active Windows/Vivado project, so treat this as a logical inventory rather than a guarantee that every file sits beside the README.

### Python/model and quantization

| File | Purpose |
|---|---|
| `mnist_mlp_best.pth` | Trained PyTorch checkpoint |
| `phase4_quantize.py` | Quantizes/evaluates the trained model and exports the integer reference, metrics, memory files, and RTL vectors |
| `quantization_params.json` | Authoritative scales, integer multiplier/shift, tensor shapes, data formats, and memory-layout metadata |
| `quantization_metrics.json` | Floating-point accuracy, integer-reference accuracy, prediction agreement, and export counts |
| `weights_fc1.mem` | FC1 signed 8-bit weights in hex-memory format |
| `biases_fc1.mem` | FC1 signed 32-bit biases in hex-memory format |
| `weights_fc2.mem` | FC2 signed 8-bit weights in hex-memory format |
| `biases_fc2.mem` | FC2 signed 32-bit biases in hex-memory format |
| `mnist_test_vectors.npz` | Python test images, labels, predictions, and integer scores |
| `integer_test_predictions.csv` | Row-wise comparison between labels, float predictions, and integer predictions |
| `rtl_test_images.mem` | Concatenated 784-byte images for RTL simulation; 100 images were exported in the reported run |
| `rtl_test_predictions.mem` | Expected integer-reference classes for RTL vectors |
| `rtl_test_labels.mem` | Ground-truth MNIST labels corresponding to exported vectors |

### Drawing and communication

| File/module | Purpose |
|---|---|
| `digit_draw_app.py` | Drawing/preprocessing front end (name used in the current workflow) |
| `sent_image.bin` | Raw 784-byte image payload |
| `send_digit_uart.py` | Sends one image payload and reads one prediction byte |
| `uart_rx.sv` | UART receiver |
| `uart_tx.sv` | UART transmitter |

### FPGA RTL / constraints / IP

| File/module | Purpose |
|---|---|
| `mlp_infer.sv` | Image buffer, MAC state machine, requantization, class-score comparison, and prediction output |
| `ml_demo_top.sv` | Top-level UART/image/inference controller and clock/reset connections |
| `clk_wiz_0` | Generated-clock IP used for the intended 50 MHz internal domain |
| `.xdc` constraint file(s) | Pin locations and clock definitions |

Keep the four memory files in the project and make sure the RTL's `$readmemh` filenames resolve to the intended files. If a memory file is regenerated, rebuild/retest the design because it changes the trained model parameters implemented in hardware.

---

## 12. Reproducing the current live test

This is a practical checklist based on the current architecture; adjust relative paths to match the actual folder layout.

### Before programming the board

1. Open the correct Vivado project and confirm the target part matches the Boolean Board.
2. Confirm `ml_demo_top.sv` is the synthesis top and that the UART modules, inference RTL, memory files, Clocking Wizard IP, and correct XDC are included.
3. Check `$readmemh` messages and confirm all four parameter memory files are read.
4. Run synthesis and implementation using the current sources.
5. Read `report_clocks` and `report_timing_summary`; do not rely on a report from an earlier run.
6. Generate the bitstream from the intended implemented design and program the board.
7. Ensure the USB/UART connection is visible as the intended COM port.

### Running the drawing-to-FPGA demo

1. Launch the drawing app.
2. Draw a digit and save/export the preprocessed image as raw `sent_image.bin` or the sender's configured path.
3. Confirm the file size is exactly 784 bytes.
4. Release reset and ensure the clock generator is locked. The design derives reset from the board reset button and the Clocking Wizard lock signal:

   ```systemverilog
   assign rst = btn[0] | ~clk_locked;
   ```

5. Run the UART sender from the appropriate directory:

   ```powershell
   python send_digit_uart.py --port COM4 --baud 115200 --image digit_output/sent_image.bin
   ```

6. Confirm it prints the number of sent bytes and one digit in the 0–9 range.
7. Repeat with several distinct images. Save every image, expected label, and FPGA prediction for later comparison.

If the sender times out, check in this order: COM port and baud rate; UART pin assignments; active bitstream; reset/clock lock; whether the image counter reaches 783; whether inference `start` pulses; whether `done` asserts; and whether the transmit state starts. If the response byte is outside 0–9, investigate framing, stale bytes, state-machine behavior, and memory/argmax logic.

---

## 13. Verification plan: what remains

The basic demo is working, but stronger verification is needed before treating the project as complete.

### A. Functional classification tests

- Draw multiple examples of each digit, not just one.
- Compare Python integer-reference predictions with physical FPGA predictions on identical 784-byte images.
- Send the 100 exported RTL vectors through the real UART interface if practical; compare each prediction against both the integer reference and MNIST label.
- Report three metrics separately: FPGA-vs-integer agreement, FPGA accuracy vs. labels, and integer-reference accuracy vs. labels.

A useful result table is:

| Sample | Ground truth | Float model | Integer reference | FPGA UART result | FPGA = integer? |
|---|---:|---:|---:|---:|---|
| 0 | … | … | … | … | … |
| 1 | … | … | … | … | … |

### B. Static timing verification

- Obtain a fresh post-implementation timing summary for the current RTL/IP/XDC snapshot.
- Verify the intended generated clock is recognized and is applied to all internal sequential logic.
- Confirm there are no clock-redefinition warnings or unexplained unconstrained paths.
- Aim for `WNS >= 0` and `TNS = 0` at the selected target clock frequency.
- Save the exact WNS/TNS, clock period, and worst-path endpoints in the project README each time a meaningful timing change is made.

### C. Resource utilization

Record post-synthesis/post-implementation LUT, FF, DSP, and block-RAM utilization. Because the weight arrays are indexed dynamically in the RTL, verify how Vivado implemented them rather than assuming they became block RAM. A memory declaration by itself does not guarantee BRAM inference; synthesis structure, read ports, and access timing affect mapping.

### D. Latency and throughput

Measure these separately:

1. Python write/flush time for 784 bytes.
2. Time until the first FPGA response byte.
3. Total transaction time.
4. Repeated-image reliability.

Use an FPGA-visible timer or a carefully measured host-side timestamp around `write` and `read` operations. Do not infer inference latency directly from total UART transaction time, because serial transfer and USB buffering are included.

### E. Protocol robustness

The first version has no framing/checksum or explicit handshake. For repeated or continuous requests, consider adding:

- a recognizable start/header sequence;
- a packet length or fixed transaction marker;
- acknowledgement/error status;
- optional image sequence ID;
- a documented behavior for receiving bytes while inference is busy;
- a defined reset/recovery sequence after a partial transfer.

These changes are not required to reproduce the current one-image demo, but they would make the interface more robust.

---

## 14. Known limitations and cautions

1. **One live prediction is not an accuracy benchmark.** The returned class 6 confirms a working transaction, not generalization across handwriting styles.
2. **The most recent timing evidence is incomplete.** The last timing report preserved here still reported 100 MHz and negative slack. A newer report after the final clock/controller edits is necessary to assert timing closure.
3. **Training remains off-chip.** The FPGA performs inference only; this is not on-board training.
4. **The transport protocol is minimal.** It sends a fixed 784-byte payload followed by a single prediction byte, without explicit packet framing or checksum.
5. **Pixel preprocessing must match training.** Image polarity, centering, intensity scaling, flattening order, or resize differences can reduce accuracy even if the RTL arithmetic is correct.
6. **Quantization constants are tied to the current checkpoint.** Regenerating weights requires regenerating memory files and updating the RTL constants consistently.
7. **Clock names do not prove clock frequency.** A signal named `clk_50mhz` is only 50 MHz if the Clocking Wizard configuration and generated-clock timing constraints actually produce and describe that frequency.
8. **Behavioral correctness and timing closure are different checks.** Simulation or one successful FPGA run cannot replace static timing analysis.

---

## 15. Recommended next milestones

### Milestone 1 — Validate the deployed model

Send a labelled set of at least 100 MNIST images through the real board UART. Compare each FPGA prediction with the integer reference and report agreement and accuracy. This verifies more than the single drawing-app example.

### Milestone 2 — Close and document timing

Confirm the top-level FSM, UART, and MLP core all use the intended clock. Resolve any remaining clock-redefinition warning. Capture a current clock summary and timing summary showing the actual generated clock frequency and final slack.

### Milestone 3 — Record hardware costs

Save LUT/FF/DSP/BRAM utilization and calculate approximate inference latency using on-board measurement. Compare the area/latency trade-offs if the multiplier or MAC architecture is changed.

### Milestone 4 — Improve the communication protocol

Add framing and a clear ready/acknowledge behavior if the goal is to run repeated predictions without manually restarting the sender or resetting the FPGA.

### Milestone 5 — Experiment with hardware architecture

Once the current implementation is functionally validated and timing reports are stable, test architectural variants such as different hidden-layer sizes, DSP-oriented MAC structures, BRAM-backed parameter storage, or limited parallelism. Compare accuracy, latency, and FPGA resource usage using the same image set.

---

## 16. Useful concepts learned in this project

- **Training vs. inference:** the PC learns weights; the FPGA evaluates a fixed learned function.
- **Quantization:** floating-point weights/activations are represented by fixed-width integers and numerical scales.
- **Multiply-accumulate (MAC):** each neuron computes weighted inputs plus bias.
- **Requantization:** converts an accumulator's numeric scale to the activation scale required by the next layer.
- **Signedness and extension:** correct signed INT8 weights, unsigned pixels, multiplication width, and sign-extension are essential to preserve the intended arithmetic.
- **Finite-state-machine control:** the sequential inference core schedules each multiply, accumulation, activation conversion, and score comparison.
- **UART:** transmits bytes serially; 8-N-1 uses ten serial bit times per byte.
- **Static timing analysis:** WNS and TNS quantify setup timing violations relative to clock constraints.
- **Critical path:** the longest constrained data path limits the maximum safe clock frequency.
- **Generated clocks:** an internal clock from a Clocking Wizard must be both configured correctly and represented correctly in timing constraints.
- **Functional validation vs. timing closure:** a design can compute a correct answer at least once and still have a timing violation that must be resolved.

---

## 17. Current status at a glance

- [x] Train the PyTorch MNIST MLP.
- [x] Evaluate the floating-point model.
- [x] Quantize the MLP and evaluate an integer reference.
- [x] Export FC1/FC2 weights and biases to memory files.
- [x] Implement sequential integer inference RTL.
- [x] Build the UART receive/transmit path.
- [x] Build the Python drawing and sending workflow.
- [x] Receive 784 image bytes on the physical FPGA and return a digit byte.
- [x] Observe a successful live prediction (`6`).
- [ ] Verify FPGA-vs-integer-reference agreement over a labelled multi-image set.
- [ ] Confirm a fresh final timing report for the final RTL and Clocking Wizard configuration.
- [ ] Record final LUT/FF/DSP/BRAM utilization.
- [ ] Measure inference latency separately from UART transfer time.
- [ ] Add packet framing/handshake if repeated streaming is a project requirement.

---

## Appendix A — Clock/timing debug checklist

Use this checklist after changing clocking or datapath RTL:

```text
[ ] Save the current RTL revision and identify it in notes.
[ ] Confirm ml_demo_top.sv is the synthesis top.
[ ] Confirm all sequential blocks in this single-clock design use clk_50mhz.
[ ] Confirm uart_rx, uart_tx, mlp_infer, and top controller use the same clock.
[ ] Confirm reset asserts when btn[0] is active or clk_locked is low.
[ ] Confirm the Clocking Wizard output is configured to 50 MHz.
[ ] Keep the root primary clock only on the external clk port (100 MHz / 10 ns).
[ ] Remove duplicate/downstream create_clock definitions.
[ ] Check the Clocking Wizard generated constraints are active.
[ ] Rerun synthesis and implementation after source, IP, or XDC changes.
[ ] Run report_clocks.
[ ] Run report_timing_summary -delay_type max.
[ ] Confirm the generated 50 MHz clock appears if 50 MHz is the target.
[ ] Confirm WNS >= 0 ns and TNS = 0 before claiming setup timing closure.
[ ] Save the final reports and the implementation run identifier.
```

## Appendix B — Minimal UART transaction checklist

```text
[ ] Board is programmed with the current bitstream.
[ ] Board clock is running; Clocking Wizard lock is asserted.
[ ] Reset button is released.
[ ] Correct COM port is selected.
[ ] Host baud is 115200, 8-N-1.
[ ] Image file exists and is exactly 784 bytes.
[ ] Image pixels use row-major order and expected polarity.
[ ] FPGA receives exactly 784 bytes.
[ ] Inference start occurs after the final pixel.
[ ] Inference done is asserted.
[ ] Prediction TX starts and sends one byte.
[ ] Host receives exactly one byte with value 0–9.
```

