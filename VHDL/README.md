# Block Diagram

# Description
## Top Module (top.vhd)
- The top-level module includes button inputs and HDMI output ports.
- It integrates several sub-modules: the NNA (Neural Network Accelerator), BRAM, and HDMI controller.
- The directional buttons (up, down, left, right) serve as player controls and are provided as input signals to the NNA module.
- The central button functions as the system start trigger.

## NNA (Neural Network Accelerator) (NNA.vhd) (4x4mac.vhd) (sigmoid.vhd)
The NNA module is used to do calculation of the neural network. It reads inputs, weight, and bias of the neural network from memory, and stores the outputs back to memory.

### Main component
- Control Unit (FSM) (Implement in NNA.vhd)
    - The control unit is implemented as a finite state machine (FSM) and is responsible for orchestrating the data flow and operation of each component within the Neural Network Accelerator (NNA).
    - The FSM includes the following states:
        - S_END: Idle state, indicating that all computations have completed.
        - S_LOAD_INFO: Loads layer information from memory into internal registers.
        - S_INFO: Decodes the loaded layer information to determine the type of neural network layer to compute. Based on the layer type, the FSM transitions to the corresponding computation state. Supported layer types include: LINEAR, MULADD, CONV3, SWAP, CONVT, and END.
		- Layer-specific states:
            - LINEAR: A sequence of states that implement a fully connected layer.
            - MULADD: A set of states that implement a layer performing the operation:
output = input × mul + input + add
            - CONV3: Implements a 3×3 convolutional layer. Input image size is 4×4, with padding = 1, resulting in an output size of 4×4. The number of input and output channels is configurable.
            - SWAP: Used to manage temporal frame data. The neural network relies on the previous 4 input frames to infer the next output. This state handles replacement of the oldest frame with the latest one during gameplay.
            - CONVT (Transposed Convolution): Implements a transposed convolution layer. Supported input image sizes include 4×4, 8×8, 16×16, and 32×32, with padding = 1 and stride = 2. The output size is twice the input size, resulting in 8×8, 16×16, 32×32, or 64×64 outputs.
- MAC: (4x4MAC.vhd)
    - The MAC (Multiply-Accumulate) unit is designed to operate with a 4×4 kernel. It receives 16 input values and 16 weight values, both in Q2.13 fixed-point format, from the input buffer and weight buffer, respectively.
    - It performs 16 multiplications, producing intermediate results in Q13.18 format. These 16 products are then summed to generate a single output, also in Q13.18 format. (When dealing with a 3x3 convolution kernel, put 0 at the last 7 weight.)
    - There are 16 4x4mac modules in the design.
- Input Channel Accumulator (Implemented in NNA.vhd)
    - The Input Channel Accumulator combines the outputs from multiple input channels to generate the final result for each output channel.
    - It contains a register array of size 64×4, with each register 32 bits wide, the value is in Q13.18 format.
        - For Linear and Convolution layers, where the output sizes are 32 neurons and 4×4 respectively, the accumulator can complete the accumulation and output process in a single loop.
        - For Transposed Convolution layers, where output sizes range from 8×8 to 64×64, accumulation cannot be completed in a single loop. Instead, the accumulator processes 4 rows at a time, gradually building the full output.
- Sigmoid (sigmoid.vhd)
    - This module implements the sigmoid activation function using a piecewise linear approximation with 16 segments. When the sigmoid enable signal is active, output will go through this module. The sigmoid function is used only at the last layer.
    - Input: Q13.18 fixed-point format (from the Input Channel Accumulator)
    - Output: Q0.15 fixed-point format There are 16 sigmoid modules in the design.

- Relu (Implemented in NNA.vhd)
    - Implement relu function. When the relu enable signal is active, output will go through the relu function. Input is Q13.18 fixed-point format (from the Input Channel Accumulator), and output is Q2.13 fixed point format.
- Input, Weight, Bias Buffer (Implemented in NNA.vhd)
    - Used to store input, weight and bias read from memory.

### Calculation of each Layer (For simplicity & clearness, use pseudocode to describe how hardware works)

- Linear
```
load bias 0~15 to input channel accumulator
load bias 16~31 to input channel accumulator

if (input neuron = 8){
    for (i=0, i<32, i++){
	load input (button) to input buffer
        load weight(i) to weight buffer
        in_ch_accu(i) <= in_ch_accu(i) + input*weight(i)
    }
}
if (input neuron = 32){
    for (i=0, i<32, i++){
        for (j=0, j<32, j=j+16){
	    load input(0+j ~ 15+j) from memory to input buffer 
            load weight(i)(0+j ~ 15+j) to weight buffer
            in_ch_accu(i) <= in_ch_accu(i) + input*weight(i)
        }
    }
}

output 32 data (2 clocks) (each clock 16 data)
```

- MULADD
```
load mul input to shift register 1 (16 data)
load add input to shift register 2 (16 data)
for (i=0, i<16, i++){
    load input to input buffer (16 data)
    in_ch_accu <= input*sft_reg1+input+sft_reg2 (16 data)
    output in_ch_accu (16 data)
    shift shift register 1&2
}
```

- CONV3 (Convolution with 3x3 Kernel)
```
(only 4x4 input size and 4x4 output size)
for (i=0, i<output channel, i++){
    load bias to in_ch_accu
    for(j=0, j<input channel, j++){
        load weight (first 8 weight) to weight buffer
        load weight (last 1 weight) to weight buffer
        load input to input buffer
        in_ch_accu <= in_ch_accu + mac_output (16 data)
    }
    output for an output channel (16 data)
}
```

- SWAP
```
read input frame(1)(0~3)
read input frame(2)(0~3)
read input frame(3)(0~3)
read result(0~3)
write input frame(1)(0~3) to input frame(0)(0~3)
write input frame(2)(0~3) to input frame(1)(0~3)
write input frame(3)(0~3) to input frame(2)(0~3)
write result(0~3) to input frame(3)(0~3)
```
- CONVT (Transposed Convolution)
```
for (i=0, i<output channel, i++){
    load bias from memory to bias buffer
    for (j=0, j<input input_size_row, j++){
        for (k=0, k<input channel, k++){
            load weight(k) to weight buffer
	    load input(k)(jth row) to input buffer
	    accumulate in_ch_accu (1 input row => 4 row in in_ch_accu)
	    (if in_size=32, each row need 2 loop)
	}
	add bias to first 2 row of in_ch_accu
	output first 2 row of in_ch_accu (output(i)(2j-1th & 2jth row))
    }
}
add bias to the last row of the last output channel
output the last row of the last output channel
```

## BRAM
We use the BRAM IP core provided by Vivado to generate the Block RAM module. Each row is configured to be 256 bits wide, with a total of 52,224 rows. A dual-port BRAM configuration is used, allowing simultaneous access by both the NNA and HDMI modules.

## HDMI
We use HDMI to display our game on a monitor at 640x480 resolution, 60 Hz refresh rate.

### HDMI Timing Generation (hdmi_timing_gen.vhd)
- Originally written in Verilog (sourced from: https://blog.csdn.net/zhoutaopower/article/details/113485579), we translated it into VHDL.
- The horizontal counter (h_counter) ranges from 0 to 799.
- The vertical counter (v_counter) ranges from 0 to 524.
- The valid display area is defined as: (h, v) = (160–799, 45–524)

### HDMI Data Generation (hdmi_data_gen.vhd)
- This module generates RGB pixel data to be displayed via HDMI.
- We display our game frame within the region: (h, v) = (256–511, 256–511)
- Our original game frame size is 64×64, and we scale it up to 256×256 for display.
- For each row of the frame, we need to read 64 RGB values (i.e., 192 values total for R, G, and B).
    - Memory data width: 16 data values per read
    - Each memory read takes 3 clock cycles
    - Total: 36 clock cycles per row
    - Data is read when h_counter is in the range 16–51
- During display:
    - If (h, v) is within (256–511, 256–511), we output the scaled frame data.
    - Otherwise, we output black.

### RGB to DVI
We use the IP core provided by Digilent.



