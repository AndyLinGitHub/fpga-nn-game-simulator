import math
from types import SimpleNamespace

import onnx
from onnx import numpy_helper, shape_inference
import numpy as np

def get_shape_from_value_info(value_info, name):
    for vi in value_info:
        if vi.name == name:
            return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    return None

def float_to_q2_13_bin(value):
    scale = 1 << 13  # 8192
    max_val = (1 << 15) - 1  # 32767
    min_val = -(1 << 15)     # -32768

    # Scale and round
    fixed_val = int(round(value * scale))

    # Clamp to 16-bit signed range
    fixed_val = max(min(fixed_val, max_val), min_val)

    # Convert to 16-bit two's complement binary string
    if fixed_val < 0:
        fixed_val = (1 << 16) + fixed_val  # two's complement

    return format(fixed_val, '016b')

def q2_13_bin_to_float(bin_str):
    if len(bin_str) != 16:
        raise ValueError("Input must be a 16-bit binary string")

    # Convert binary string to signed integer
    val = int(bin_str, 2)
    if val & (1 << 15):  # check sign bit
        val -= (1 << 16)  # convert from two's complement

    # Scale back to float
    return val / (1 << 13)

def array_to_binary_str(array, ml=256):
    binarys = []
    binary_str = ""
    for i in range(len(array)):
        value = array[i]
        binary_value = float_to_q2_13_bin(value)
        binary_str = binary_value + binary_str

        if len(binary_str) == ml:
            binary_str = (256 - len(binary_str))*"0" + binary_str
            binarys.append(binary_str)
            binary_str = ""
        elif i == len(array) - 1:
            binary_str = (256 - len(binary_str))*"0" + binary_str
            binarys.append(binary_str)
            binary_str = ""

    return binarys

def initial_bram():
    bram = ["0" * 256 for _ in range(max_address+1)]
    frame_input = np.load('initial_input.npy')
    frame_input_binary_str = array_to_binary_str(frame_input.flatten())

    for i in range(len(frame_input_binary_str)):
        default = bram[preserve_block_0[0]+i]
        assert default != "0"*16
        bram[preserve_block_0[0]+i] = frame_input_binary_str[i]

    counter = 0
    for k, v in layer_info.items():
        binary_layer_type = format(v["layer_type"], f'0{4}b')
        binary_in_size = format(v["in_size"], f'0{9}b')
        binary_in_channel = format(v["in_channel"], f'0{9}b')
        binary_out_channel = format(v["out_channel"], f'0{9}b')
        binary_relu_enable = format(v["relu_enable"], f'0{1}b')
        binary_sigmoid_enable = format(v["sigmoid_enable"], f'0{1}b')

        binary_str = binary_sigmoid_enable + binary_layer_type + binary_in_size + binary_in_channel + binary_out_channel + binary_relu_enable

        binary_weight_address = format(v["weight_address"], f'0{16}b')
        binary_bias_address = format(v["bias_address"], f'0{16}b')
        binary_input_address_0 = format(v["input_address_0"], f'0{16}b')
        binary_input_address_1 = format(v["input_address_1"], f'0{16}b')
        binary_output_address = format(v["output_address"], f'0{16}b')

        binary_str += binary_weight_address + binary_bias_address + binary_input_address_0 + binary_input_address_1 + binary_output_address
        binary_str = (256 - len(binary_str))*"0" + binary_str
        bram[counter] = binary_str
        counter += 1

        if name_to_nodes[k].op_type == "Gemm":
            array_weight = initializer_map[v["weight"]].flatten()
            array_bias = initializer_map[v["bias"]].flatten()
            weight_binary_str = array_to_binary_str(array_weight)
            bias_binary_str = array_to_binary_str(array_bias)

            for i in range(len(weight_binary_str)):
                default = bram[v["weight_address"]+i]
                assert default != "0"*16
                bram[v["weight_address"]+i] = weight_binary_str[i]
            
            for i in range(len(bias_binary_str)):
                default = bram[v["bias_address"]+i]
                assert default != "0"*16
                bram[v["bias_address"]+i] = bias_binary_str[i]

        elif name_to_nodes[k].op_type == "Add" or name_to_nodes[k].op_type == "output_inplace" or name_to_nodes[k].op_type == "end":
            pass

        elif name_to_nodes[k].op_type == "Conv":
            array_weight = initializer_map[v["weight"]]
            array_weight = array_weight.reshape(array_weight.shape[0], array_weight.shape[1], -1)
            array_weight_f = array_weight[:, :, :array_weight.shape[-1]-1].flatten()  # First k*k - 1 elements along the last dimension
            array_weight_1 = array_weight[:, :, array_weight.shape[-1]-1:].flatten()  # Last element along the last dimension

            weight_binary_str = array_to_binary_str(array_weight_f)
            weight_binary_str_1 = array_to_binary_str(array_weight_1)
            array_bias = initializer_map[v["bias"]].flatten()
            bias_binary_str = array_to_binary_str(array_bias)

            for i in range(len(weight_binary_str)):
                default = bram[v["weight_address"]+i]
                assert default != "0"*16
                bram[v["weight_address"]+i] = weight_binary_str[i]

            for i in range(len(weight_binary_str_1)):
                default = bram[v["input_address_1"]+i]
                assert default != "0"*16
                bram[v["input_address_1"]+i] = weight_binary_str_1[i]
            
            for i in range(len(bias_binary_str)):
                default = bram[v["bias_address"]+i]
                assert default != "0"*16
                bram[v["bias_address"]+i] = bias_binary_str[i]

            array_bias = initializer_map[v["bias"]].flatten()

        elif name_to_nodes[k].op_type == "ConvTranspose":
            array_weight = np.transpose(initializer_map[v["weight"]], (1, 0, 2, 3)).flatten()
            array_bias = initializer_map[v["bias"]].flatten()
            weight_binary_str = array_to_binary_str(array_weight)
            bias_binary_str = array_to_binary_str(array_bias)

            for i in range(len(weight_binary_str)):
                default = bram[v["weight_address"]+i]
                assert default != "0"*16
                bram[v["weight_address"]+i] = weight_binary_str[i]
            
            for i in range(len(bias_binary_str)):
                default = bram[v["bias_address"]+i]
                assert default != "0"*16
                bram[v["bias_address"]+i] = bias_binary_str[i]
            
        else:
            assert False

    return bram

# Load and infer shapes
model = onnx.load("model.onnx")
model = shape_inference.infer_shapes(model)
graph = model.graph

# Build name → shape lookup
value_info = list(graph.value_info) + list(graph.input) + list(graph.output)
name_to_shape = {vi.name: get_shape_from_value_info(value_info, vi.name) for vi in value_info}

# Build initializer dict
initializer_map = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

name_to_nodes = {node.name: node for node in graph.node}

layer_info = {
"/conv_model/film/projector/projector.0/Gemm": {
    "input": ["first_button_encoding", ""],
    "weight": "conv_model.film.projector.0.weight",
    "bias": "conv_model.film.projector.0.bias",
    "layer_type": 0,
    "in_channel": name_to_shape.get(name_to_nodes["/conv_model/film/projector/projector.0/Gemm"].input[0])[-1],
    "out_channel": name_to_shape.get(name_to_nodes["/conv_model/film/projector/projector.0/Gemm"].output[0])[-1],
    "in_size": 1,
    "relu_enable": 1,
    "output_block": 1,
    "sigmoid_enable": 0,
    },

"/conv_model/film/projector/projector.2/Gemm": {
    "input": ["/conv_model/film/projector/projector.0/Gemm", ""],
    "weight": "conv_model.film.projector.2.weight",
    "bias": "conv_model.film.projector.2.bias",
    "layer_type": 0,
    "in_channel": name_to_shape.get(name_to_nodes["/conv_model/film/projector/projector.2/Gemm"].input[0])[-1],
    "out_channel": name_to_shape.get(name_to_nodes["/conv_model/film/projector/projector.2/Gemm"].output[0])[-1],
    "in_size": 1,
    "relu_enable": 0,
    "output_block": 2,
    "sigmoid_enable": 0,
    },

# Fuse of rest of film block
"/conv_model/film/Add_2": {
    "input": ["first_input_frame", "/conv_model/film/projector/projector.2/Gemm"],
    "weight": None,
    "bias": None,
    "layer_type": 1,
    "in_channel": name_to_shape["input"][1],
    "out_channel": name_to_shape["input"][1],
    "in_size": name_to_shape["input"][-1],
    "relu_enable": 0,
    "output_block": 1,
    "sigmoid_enable": 0,
    },

# First Conv
"/conv_model/unet/in_conv/conv/conv.0/Conv": {
    "input": ["/conv_model/film/Add_2", ""],
    "weight": "conv_model.unet.in_conv.conv.0.weight",
    "bias": "conv_model.unet.in_conv.conv.0.bias",
    "layer_type": 2,
    "in_channel": name_to_shape["/conv_model/Concat_output_0"][1],
    "out_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.0/Conv_output_0"][1],
    "in_size": name_to_shape["/conv_model/Concat_output_0"][-1],
    "relu_enable": 1,
    "output_block": 2,
    "sigmoid_enable": 0,
    },

# 2 Conv
"/conv_model/unet/in_conv/conv/conv.2/Conv": {
    "input": ["/conv_model/unet/in_conv/conv/conv.0/Conv", ""],
    "weight": "conv_model.unet.in_conv.conv.2.weight",
    "bias": "conv_model.unet.in_conv.conv.2.bias",
    "layer_type": 2,
    "in_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.0/Conv_output_0"][1],
    "out_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.2/Conv_output_0"][1],
    "in_size": name_to_shape["/conv_model/unet/in_conv/conv/conv.0/Conv_output_0"][-1],
    "relu_enable": 1,
    "output_block": 3,
    "sigmoid_enable": 0,
    },

# 4 Conv
"/conv_model/unet/in_conv/conv/conv.4/Conv": {
    "input": ["/conv_model/unet/in_conv/conv/conv.2/Conv", ""],
    "weight": "conv_model.unet.in_conv.conv.4.weight",
    "bias": "conv_model.unet.in_conv.conv.4.bias",
    "layer_type": 2,
    "in_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.2/Conv_output_0"][1],
    "out_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.4/Conv_output_0"][1],
    "in_size": name_to_shape["/conv_model/unet/in_conv/conv/conv.2/Conv_output_0"][-1],
    "relu_enable": 1,
    "output_block": 2,
    "sigmoid_enable": 0,
    },

# 6 Conv
"/conv_model/unet/in_conv/conv/conv.6/Conv": {
    "input": ["/conv_model/unet/in_conv/conv/conv.4/Conv", ""],
    "weight": "conv_model.unet.in_conv.conv.6.weight",
    "bias": "conv_model.unet.in_conv.conv.6.bias",
    "layer_type": 2,
    "in_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.4/Conv_output_0"][1],
    "out_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.6/Conv_output_0"][1],
    "in_size": name_to_shape["/conv_model/unet/in_conv/conv/conv.4/Conv_output_0"][-1],
    "relu_enable": 1,
    "output_block": 3,
    "sigmoid_enable": 0,
    },

# out Conv
"/conv_model/unet/out_conv/Conv": {
    "input": ["/conv_model/unet/in_conv/conv/conv.6/Conv", ""],
    "weight": "conv_model.unet.out_conv.weight",
    "bias": "conv_model.unet.out_conv.bias",
    "layer_type": 2,
    "in_channel": name_to_shape["/conv_model/unet/in_conv/conv/conv.6/Conv_output_0"][1],
    "out_channel": name_to_shape["/conv_model/unet/out_conv/Conv_output_0"][1],
    "in_size": name_to_shape["/conv_model/unet/in_conv/conv/conv.6/Conv_output_0"][-1],
    "relu_enable": 0,
    "output_block": 2,
    "sigmoid_enable": 0,
    },

"output_inplace": {
"input": ["/conv_model/unet/out_conv/Conv", ""],
"weight": None,
"bias": None,
"layer_type": 3,
"in_channel": name_to_shape["/conv_model/unet/out_conv/Conv_output_0"][1],
"out_channel": name_to_shape["input"][1],
"in_size": name_to_shape["/conv_model/unet/out_conv/Conv_output_0"][-1],
"relu_enable": 0,
"output_block": 0,
"sigmoid_enable": 0,
    },

"/vae_decoder/convt0/ConvTranspose": {
    "input": ["/conv_model/unet/out_conv/Conv", ""],
    "weight": "vae_decoder.convt0.weight",
    "bias": "vae_decoder.convt0.bias",
    "layer_type": 4,
    "in_channel": name_to_shape["/conv_model/unet/out_conv/Conv_output_0"][1],
    "out_channel": name_to_shape["/vae_decoder/convt0/ConvTranspose_output_0"][1],
    "in_size": name_to_shape["/conv_model/unet/out_conv/Conv_output_0"][-1],
    "relu_enable": 1,
    "output_block": 3,
    "sigmoid_enable": 0,
    },

"/vae_decoder/convt1/ConvTranspose": {
    "input": ["/vae_decoder/convt0/ConvTranspose", ""],
    "weight": "vae_decoder.convt1.weight",
    "bias": "vae_decoder.convt1.bias",
    "layer_type": 4,
    "in_channel": name_to_shape["/vae_decoder/convt0/ConvTranspose_output_0"][1],
    "out_channel": name_to_shape["/vae_decoder/convt1/ConvTranspose_output_0"][1],
    "in_size": name_to_shape["/vae_decoder/convt0/ConvTranspose_output_0"][-1],
    "relu_enable": 1,
    "output_block": 2,
    "sigmoid_enable": 0,
    },

"/vae_decoder/convt2/ConvTranspose": {
    "input": ["/vae_decoder/convt1/ConvTranspose", ""],
    "weight": "vae_decoder.convt2.weight",
    "bias": "vae_decoder.convt2.bias",
    "layer_type": 4,
    "in_channel": name_to_shape["/vae_decoder/convt1/ConvTranspose_output_0"][1],
    "out_channel": name_to_shape["/vae_decoder/convt2/ConvTranspose_output_0"][1],
    "in_size": name_to_shape["/vae_decoder/convt1/ConvTranspose_output_0"][-1],
    "relu_enable": 1,
    "output_block": 3,
    "sigmoid_enable": 0,
    },

"/vae_decoder/convt3/ConvTranspose": {
    "input": ["/vae_decoder/convt2/ConvTranspose", ""],
    "weight": "vae_decoder.convt3.weight",
    "bias": "vae_decoder.convt3.bias",
    "layer_type": 4,
    "in_channel": name_to_shape["/vae_decoder/convt2/ConvTranspose_output_0"][1],
    "out_channel": name_to_shape["/vae_decoder/convt3/ConvTranspose_output_0"][1],
    "in_size": name_to_shape["/vae_decoder/convt2/ConvTranspose_output_0"][-1],
    "relu_enable": 0,
    "output_block": 4,
    "sigmoid_enable": 1,
    }, 

"end": {
    "input": ["/vae_decoder/convt3/ConvTranspose", ""],
    "weight": None,
    "bias": None,
    "layer_type": 5,
    "in_channel": 3,
    "out_channel": 3,
    "in_size": 64,
    "relu_enable": 0,
    "output_block": 4,
    "sigmoid_enable": 0,
    }
}

name_to_nodes["output_inplace"] =  SimpleNamespace(op_type="output_inplace") 
name_to_nodes["end"] =  SimpleNamespace(op_type="end") 

max_weight_num = 32768
max_address = 52223
preserve_block_4 = [max_address - int(np.ceil(64*64*3*16/256)) + 1, max_address] # [start, end] (include)
preserve_block_3 = [preserve_block_4[0] - 1 - int(np.ceil(max_weight_num*16/256)) + 1, max_address] # [start, end] (include)
preserve_block_2 = [preserve_block_3[0] - 1 - int(np.ceil(max_weight_num/2*16/256)) + 1, preserve_block_3[0] - 1]
preserve_block_0 = [preserve_block_2[0] - 1 - int(np.ceil(256*16/256)) + 1, preserve_block_2[0] - 1]
preserve_block_1 = [preserve_block_0[0] - 1 - int(np.ceil(256*16/256)) + 1, preserve_block_0[0] - 1]

output_addresses = [preserve_block_0, preserve_block_1, preserve_block_2, preserve_block_3, preserve_block_4]

weight_address = len(layer_info)
for k, v in layer_info.items():
    input_addresses = []
    for i in v["input"]:
        if "first_button_encoding" == i:
            input_address = 2**16 - 1
        elif "first_input_frame" == i:
            input_address = preserve_block_0[0]
        elif len(i) > 0:
            input_address = layer_info[i]["output_address"]
        else:
            input_address = 2**16 - 1

        input_addresses.append(input_address)
    
    layer_info[k]["input_address_0"] = input_addresses[0]
    layer_info[k]["input_address_1"] = input_addresses[1]

    layer_info[k]["output_address"] = output_addresses[layer_info[k]["output_block"]][0]

    if layer_info[k]["weight"] is not None and layer_info[k]["bias"] is not None:
        weight = initializer_map[layer_info[k]["weight"]]
        weight_size = math.prod(weight.shape)
        weight_row_num = int(np.ceil(weight_size*16/256))

        if name_to_nodes[k].op_type == "Conv":
            kernel_offset_num = int(np.ceil(math.prod(weight.shape[:2])*16/256))

        bias = initializer_map[layer_info[k]["bias"]]
        bias_size = math.prod(bias.shape)
        bias_row_num = int(np.ceil(bias_size*16/256))

        layer_info[k]["weight_address"] = weight_address

        bias_address = weight_address + weight_row_num
        layer_info[k]["bias_address"] = bias_address
        
        if name_to_nodes[k].op_type == "Conv":
            layer_info[k]["input_address_1"] = weight_address + weight_row_num - kernel_offset_num
            
        weight_address = bias_address + bias_row_num

        

        assert bias_address < preserve_block_1[0]
    else:
        layer_info[k]["weight_address"] = 2**16 - 1
        layer_info[k]["bias_address"] = 2**16 - 1

bram = initial_bram()
hex_bram = [format(int(b, 2), '064X') for b in bram]

with open('initial_hex_bram.coe', 'w') as f:
    f.write("memory_initialization_radix = 16;" + '\n')
    f.write("memory_initialization_vector = " + '\n')
    for hex_str in hex_bram:
        f.write(hex_str + ', \n')
    f.write(";")