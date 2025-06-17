# Neural Network-Based Game Simulator on FPGA
## Description
- This project implements a neural network–based game simulator on an FPGA, which uses a neural network to predict future game frames based on past predictions and player inputs from FPGA buttons while simultaneously outputting the frames to a display via HDMI.
- In this implementation, a neural network is trained on the Pong game. Once trained, the model's weights and layer configurations are converted into rows of hexadecimal values, which are used to initialize the FPGA's Block RAM during bitstream generation.
- While the FPGA is running, the Neural Network Accelerator implemented on the FPGA reads layer information, weights, and input feature maps from Block RAM, computes the output feature map, and writes it back to Block RAM. Simultaneously, the HDMI module reads the predicted frame from Block RAM and outputs it to the display via the FPGA’s HDMI port.
- With a fixed neural network architecture, the game simulated on the FPGA can be changed simply by updating the layer information and weights initialized in Block RAM, enabling a seamless, hardware-integrated, machine learning–driven game simulation.
- TODO: Add GIF

## Requirements
- Nexys Video
- Vivado Installation with Xilinx SDK
- HDMI Cable
- HDMI capable Monitor/Display

## Demo Setup
- Create a new Vivado project and set the target language to VHDL. Then, add all the vhdl files from the ```VHDL``` folder as source files to the project.
![Step 1](Image/step_1.png)
- On the following page, add the constraint file located in the ```VHDL``` folder. Then, in the subsequent steps, select the Nexys Video board as the target hardware. Complete the remaining steps to finish creating the project.
- Download the HDMI example provided by Digilent in the following repository: ```https://github.com/Digilent/Nexys-Video-HDMI#```. Then, add the ```rgb2dvi``` IP core from this repository to your Vivado project.
![Step 3](Image/step_3.png)






