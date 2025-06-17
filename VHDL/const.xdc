##Clock Signal
set_property -dict { PACKAGE_PIN R4    IOSTANDARD LVCMOS33 } [get_ports { clock }];
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports {clock}];
set_property -dict { PACKAGE_PIN G4   IOSTANDARD LVCMOS33} [get_ports { reset }]; #IO_L12N_T1_MRCC_35 Sch=cpu_resetn

##KEY
set_property -dict { PACKAGE_PIN F15  IOSTANDARD LVCMOS33} [get_ports {button_u }]; #IO_L20N_T3_16 Sch=btnc
set_property -dict { PACKAGE_PIN D22  IOSTANDARD LVCMOS33} [get_ports { button_d  }]; #IO_L22N_T3_16 Sch=btnd
set_property -dict { PACKAGE_PIN C22  IOSTANDARD LVCMOS33} [get_ports { button_l  }]; #IO_L20P_T3_16 Sch=btnl
set_property -dict { PACKAGE_PIN D14   IOSTANDARD LVCMOS33} [get_ports { button_r }]; #IO_L12N_T1_MRCC_35 Sch=cpu_resetn
set_property -dict { PACKAGE_PIN B22   IOSTANDARD LVCMOS33} [get_ports { button_c }]; #IO_L12N_T1_MRCC_35 Sch=cpu_resetn

## HDMI
set_property -dict { PACKAGE_PIN U1    IOSTANDARD LVDS     } [get_ports { TMDS_clk_n }]; #IO_L1N_T0_34 Sch=hdmi_tx_clk_n
set_property -dict { PACKAGE_PIN T1    IOSTANDARD LVDS     } [get_ports { TMDS_clk_p }]; #IO_L1P_T0_34 Sch=hdmi_tx_clk_p
set_property -dict { PACKAGE_PIN Y1    IOSTANDARD LVDS     } [get_ports { TMDS_data_n[0] }]; #IO_L5N_T0_34 Sch=hdmi_tx_n[0]
set_property -dict { PACKAGE_PIN W1    IOSTANDARD LVDS     } [get_ports { TMDS_data_p[0] }]; #IO_L5P_T0_34 Sch=hdmi_tx_p[0]
set_property -dict { PACKAGE_PIN AB1   IOSTANDARD LVDS     } [get_ports { TMDS_data_n[1] }]; #IO_L7N_T1_34 Sch=hdmi_tx_n[1]
set_property -dict { PACKAGE_PIN AA1   IOSTANDARD LVDS     } [get_ports { TMDS_data_p[1] }]; #IO_L7P_T1_34 Sch=hdmi_tx_p[1]
set_property -dict { PACKAGE_PIN AB2   IOSTANDARD LVDS     } [get_ports { TMDS_data_n[2] }]; #IO_L8N_T1_34 Sch=hdmi_tx_n[2]
set_property -dict { PACKAGE_PIN AB3   IOSTANDARD LVDS     } [get_ports { TMDS_data_p[2] }]; #IO_L8P_T1_34 Sch=hdmi_tx_p[2]