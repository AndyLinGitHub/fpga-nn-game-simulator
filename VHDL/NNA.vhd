--------------------------------------------------------------------------------
-- This module implement the whole Neural Network

-- For detail explanation, reference to documentation
-- It is hard to explain only through comment

-- input:
--  clock: clock
--  reset: active low reset
--  start: signal tell NNA to start calculate for a frame
--  button: 8 16bit(Q2.13) signal from button (value only have 0 or 1)
--  mem_in: data read from memory

-- output:
--  mem_en: memory enable signal
--  mem_RW: read:'0', write:'1'
--  mem_out: data write to memory
--  mem_addr: memory address
--  finish: indicate NNA finish calculating a frame

-- Memory: 256 bits data width (16 16bits data each time)

-- The design have a Finite State Machine
--  S_END: IDLE state
--  S_LOAD_INFO: load layer information from memory
--  S_INFO: decode layer information, decide which type of layer is going to calculate 
--  MULADD: this layer include multiplication and addition of metrix
--      S_MULADD_mul: load multiplication input from memory
--      S_MULADD_add: load addition input from memory
--      S_MULADD_input: load input from memory
--      S_MULADD_ex: execution state
--      S_MULADD_out: write output to memory
--  FULLY CONNECTED LAYER: this layer calculate a fully connected layer
--      S_LINEAR_bias: load bias to input channel accumulator
--      S_LINEAR_input: load input to input buffer
--      S_LINEAR_wgt: load weight to weight buffer
--      S_LINEAR_EX: execution state, add kernel output to input channel accumulator
--      S_LINEAR_out: write output to memory
--  CONV3: this layer calculate 3x3 kernel convolution with inputize 4x4(padding=1)
--      S_CONV3_bias: load bias to input channel accumulator for each output channel
--      S_CONV3_wgt1: load first 8 weight of 3x3kernel to weight buffer for each input channel
--      S_CONV3_wgt2: load last  1 weight of 3x3kernel to weight buffer for each input channel
--      S_CONV3_input: load 4x4 input to input buffer
--      S_CONV3_ex: execution state, add kernel output to input channel accumulator
--      S_CONV3_out: write output to memory (1 output channel each loop)
--  SWAP: replace the oldest frame by calculated latest frame
--      S_SWAP_input: load input with correct sequence
--      S_SWAP_out: write back to memory
--  CONVT: calculate transposed convolution (have 4x4, 8x8, 16x16, 32x32 input size)
--          (calculate part of output channel and output 2 row in each loop)
--      S_CONVT_bias: load bias to bias buffer for each output channel
--      S_CONVT_wgt: load weight to weight buffer for each input channel
--      S_CONVT_input: load input to input buffer
--      S_CONVT_ex: execution state, add kernel output to input channel accumulator
--      S_CONVT_addbias: add bias to input channel accumulator that will be output at next state
--      S_CONVT_out: write output to memory, it will output 2 row of image
--      S_CONVT_endbias: the last output row of this layer
--      S_CONVT_endout:  the last output row of this layer



-- Submodule:
--  mac: calculate multiplication & MAC of 16 input x 16 weight, have 16 mac module in the design
--          we reuse these macs in calculation of different layer
--  sigmoid: calculate sigmouid function: have 16 sigmoid module in the design

-- design by Tzu-Chi Huang
--------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity NNA  is
    port(
        clock   : in  std_logic;
        reset   : in  std_logic;
        start   : in  std_logic;
        button  : in  std_logic_vector( 8*16-1 downto 0);
        mem_in  : in  std_logic_vector(16*16-1 downto 0);
        mem_en  : out std_logic;
        mem_RW  : out std_logic;
        mem_out : out std_logic_vector(16*16-1 downto 0);
        mem_addr: out std_logic_vector(15 downto 0);
        finish  : out std_logic
    );

end  NNA;

architecture RTL  of  NNA  is
    constant read  : std_logic := '0';
    constant write : std_logic := '1';

    component mac4x4 is
    Port (in0  : in  signed (15 downto 0); --Q2.13 input
          in1  : in  signed (15 downto 0); --Q2.13 input
          in2  : in  signed (15 downto 0); --Q2.13 input
          in3  : in  signed (15 downto 0); --Q2.13 input
          in4  : in  signed (15 downto 0); --Q2.13 input
          in5  : in  signed (15 downto 0); --Q2.13 input
          in6  : in  signed (15 downto 0); --Q2.13 input
          in7  : in  signed (15 downto 0); --Q2.13 input
          in8  : in  signed (15 downto 0); --Q2.13 input
          in9  : in  signed (15 downto 0); --Q2.13 input
          in10 : in  signed (15 downto 0); --Q2.13 input
          in11 : in  signed (15 downto 0); --Q2.13 input
          in12 : in  signed (15 downto 0); --Q2.13 input
          in13 : in  signed (15 downto 0); --Q2.13 input
          in14 : in  signed (15 downto 0); --Q2.13 input
          in15 : in  signed (15 downto 0); --Q2.13 input
          wgt0  : in  signed (15 downto 0); --Q2.13 weight
          wgt1  : in  signed (15 downto 0); --Q2.13 weight
          wgt2  : in  signed (15 downto 0); --Q2.13 weight
          wgt3  : in  signed (15 downto 0); --Q2.13 weight
          wgt4  : in  signed (15 downto 0); --Q2.13 weight
          wgt5  : in  signed (15 downto 0); --Q2.13 weight
          wgt6  : in  signed (15 downto 0); --Q2.13 weight
          wgt7  : in  signed (15 downto 0); --Q2.13 weight
          wgt8  : in  signed (15 downto 0); --Q2.13 weight
          wgt9  : in  signed (15 downto 0); --Q2.13 weight
          wgt10 : in  signed (15 downto 0); --Q2.13 weight
          wgt11 : in  signed (15 downto 0); --Q2.13 weight
          wgt12 : in  signed (15 downto 0); --Q2.13 weight
          wgt13 : in  signed (15 downto 0); --Q2.13 weight
          wgt14 : in  signed (15 downto 0); --Q2.13 weight
          wgt15 : in  signed (15 downto 0); --Q2.13 weight
          mul0  : out signed(31 downto 0); --Q13.18 multiplication output
          mul1  : out signed(31 downto 0); --Q13.18 multiplication output
          mul2  : out signed(31 downto 0); --Q13.18 multiplication output
          mul3  : out signed(31 downto 0); --Q13.18 multiplication output
          mul4  : out signed(31 downto 0); --Q13.18 multiplication output
          mul5  : out signed(31 downto 0); --Q13.18 multiplication output
          mul6  : out signed(31 downto 0); --Q13.18 multiplication output
          mul7  : out signed(31 downto 0); --Q13.18 multiplication output
          mul8  : out signed(31 downto 0); --Q13.18 multiplication output
          mul9  : out signed(31 downto 0); --Q13.18 multiplication output
          mul10 : out signed(31 downto 0); --Q13.18 multiplication output
          mul11 : out signed(31 downto 0); --Q13.18 multiplication output
          mul12 : out signed(31 downto 0); --Q13.18 multiplication output
          mul13 : out signed(31 downto 0); --Q13.18 multiplication output
          mul14 : out signed(31 downto 0); --Q13.18 multiplication output
          mul15 : out signed(31 downto 0); --Q13.18 multiplication output
          MAC : out  signed (31 downto 0)); --Q13.18 sum of multiplication
    end component;

    component sigmoid is
    Port (in0  : in  signed(31 downto 0); --Q13.18
          out0 : out signed(15 downto 0)); --Q0.15
    end component;

    -- memory address
    signal mem_addr_0 : unsigned(15 downto 0);

    -- memory read counter (memory read will delay by 2 clocks)
    signal mem_read_cnt, mem_read_cnt_nxt : unsigned(1 downto 0);

    -- FSM
    type state_type is (S_END, S_LOAD_INFO, S_INFO, 
                        S_MULADD_mul, S_MULADD_add, S_MULADD_input, S_MULADD_ex, S_MULADD_out,
                        S_CONV3_bias, S_CONV3_wgt1, S_CONV3_wgt2, S_CONV3_input, S_CONV3_ex, S_CONV3_out, 
                        S_LINEAR_bias, S_LINEAR_input, S_LINEAR_wgt, S_LINEAR_EX, S_LINEAR_out,
                        S_SWAP_input, S_SWAP_out,
                        S_CONVT_bias, S_CONVT_wgt, S_CONVT_input, S_CONVT_ex, S_CONVT_addbias, S_CONVT_out, S_CONVT_endbias, S_CONVT_endout); -- FSM
    signal state : state_type; -- FSM

    -- info counter, count which layer we are calculating
    signal info_cnt : unsigned(3 downto 0);

    -- register to store layer information
    signal layer_info, layer_info_nxt : std_logic_vector(112 downto 0);
    signal load_layer_info : std_logic;

    -- layer information decoding
    signal layer_type : std_logic_vector(3 downto 0);
    constant L_linear : std_logic_vector(3 downto 0) := "0000";
    constant L_muladd : std_logic_vector(3 downto 0) := "0001";
    constant L_conv3  : std_logic_vector(3 downto 0) := "0010";
    constant L_swap   : std_logic_vector(3 downto 0) := "0011";
    constant L_convT  : std_logic_vector(3 downto 0) := "0100";
    constant L_END    : std_logic_vector(3 downto 0) := "0101";

    -- layer information decoding
    signal sigmoid_en  : std_logic; -- whether pass through sigmoid function before write to memory
    signal in_size     : unsigned(8 downto 0); -- input size of the layer
    signal in_channel  : unsigned(8 downto 0); -- input channel of the layer
    signal out_channel : unsigned(8 downto 0); -- output channel of the layer
    signal relu_en     : std_logic; -- whether pass through relu function before write to memory
    signal wgt_addr    : unsigned(15 downto 0); -- starting address of storing weight of the layer
    signal bias_addr   : unsigned(15 downto 0); -- starting address of storing bias of the layer
    signal in_addr    : unsigned(15 downto 0);  -- starting address of storing input of the layer
    signal in_wgt_addr2    : unsigned(15 downto 0); --starting address of storing extra input or weight of the layer
    signal out_addr    : unsigned(15 downto 0); -- starting address of output of the layer

    -- input channel accumulator (4x64x32bit)
    type array_4x64x32 is array (0 to 255) of signed(31 downto 0);
    signal in_ch_accu, in_ch_accu_nxt : array_4x64x32;

    type array_16x16 is array (0 to 15) of signed(15 downto 0);
    signal input_buf, input_buf_nxt : array_16x16;  -- input buffer
    signal wgt_buf, wgt_buf_nxt : array_16x16;      -- weight buffer
    signal bias_buf1, bias_buf1_nxt, bias_buf2, bias_buf2_nxt: signed(31 downto 0); -- bias buffer (only used for transposed convolution)

    -- shift register used to calculate MULADD layer
    signal shf_reg1, shf_reg1_nxt, shf_reg2, shf_reg2_nxt : array_16x16;

    signal in_ch_cnt, in_ch_cnt_nxt : unsigned(8 downto 0);     -- input channel counter
    signal out_ch_cnt, out_ch_cnt_nxt : unsigned(8 downto 0);   -- output channel counter
    signal ch_cnt, ch_cnt_nxt : unsigned(14 downto 0);          -- channel counter
    signal rowcol_cnt, rowcol_cnt_nxt : unsigned(11 downto 0);  -- row&column counter
    signal out_rowcol_cnt, out_rowcol_cnt_nxt : unsigned(11 downto 0);  -- row&column counter for output state

    -- 16-1 multiplexer to select data from memory
    signal out_ch_cnt_mod16, ch_cnt_mod16 : integer range 15 downto 0;
    signal mem_in_sel16_out_ch: signed(31 downto 0);


    type array_16x32 is array (0 to 15) of signed(31 downto 0);
    type array_16x16x16 is array (0 to 15) of array_16x16;
    type array_16x16x32 is array (0 to 15) of array_16x32;
    signal mac_in : array_16x16x16; -- mac input
    signal mac_wgt: array_16x16x16; -- mac weight input
    signal mac_mul: array_16x16x32; -- mac multiplication output
    signal mac_out : array_16x32;   -- mac mac output

    signal muladd_mul : array_16x32; -- multiplication result of MULADD layer

    signal sigmoid_out : array_16x16; -- sigmoid function output
begin
    
    -- NNA is not calculating
    finish <= '1' when (state=S_END) else '0';

    -- assign memory address
    mem_addr <= std_logic_vector(mem_addr_0);

    -- FSM & info_cnt ---------------------------------------------------------------------
    process(clock)
    begin
        if (rising_edge(clock)) then
            if (reset = '0') then
                state <= S_END;
                info_cnt <= (others => '0');
            else
                state <= state;
                info_cnt <= info_cnt;
                case (state) is
                    when S_LOAD_INFO => -- load layer information from memory
                        if (mem_read_cnt=2) then -- takes 2 clock to read
                            state <= S_INFO;
                        end if;
                    when S_INFO =>  -- decode layer information
                        case (layer_type) is
                            when L_linear =>
                                state <= S_LINEAR_bias;
                            when L_muladd =>
                                state <= S_MULADD_mul;
                            when L_conv3 =>
                                state <= S_CONV3_bias;
                            when L_swap =>
                                state <= S_SWAP_input;
                            when L_convT =>
                                state <= S_CONVT_bias;
                            when L_END =>
                                state <= S_END;
                            when others =>
                                state <= S_END;
                        end case;
                    when S_MULADD_mul => -- load multiplication source
                        if (mem_read_cnt=2) then
                            state <= S_MULADD_add;
                        end if;
                    when S_MULADD_add => -- load addition source
                        if (mem_read_cnt=2) then
                            state <= S_MULADD_input;
                        end if;
                    when S_MULADD_input =>  -- load input
                        if (mem_read_cnt=2) then
                            state <= S_MULADD_ex;
                        end if;
                    when S_MULADD_ex => -- execution state
                        state <= S_MULADD_out;
                    when S_MULADD_out =>    -- write output to memory
                        -- loop for 16 times
                        if (out_ch_cnt=15) then
                            state <= S_LOAD_INFO;
                            info_cnt <= info_cnt+1;
                        else
                            state <= S_MULADD_input;
                        end if;
                    when S_CONV3_bias => -- load bias
                        if (mem_read_cnt=2) then
                            state <= S_CONV3_wgt1;
                        end if;
                    when S_CONV3_wgt1 => -- load first 8 weight
                        if (mem_read_cnt=2) then
                            state <= S_CONV3_wgt2;
                        end if;
                    when S_CONV3_wgt2 => -- load last 1 weight
                        state <= S_CONV3_input;
                    when S_CONV3_input => -- load 4x4 input frame
                        state <= S_CONV3_ex;
                    when S_CONV3_ex => -- execution state
                        -- caltulate and accumulate input channel,
                        -- and go back to wgt state to calculate next input channel
                        -- after looping for all input channel, go to output state
                        if (in_ch_cnt=in_channel-1) then
                            state <= S_CONV3_out;
                        else
                            state <= S_CONV3_wgt1;
                        end if;
                    when S_CONV3_out => -- output state
                        -- output for 1 output channel, 
                        -- and go back to bias state to calulate next output channel loop
                        -- after looping all output channel, finish this layer
                        if (out_ch_cnt=out_channel-1) then
                            state <= S_LOAD_INFO;
                            info_cnt <= info_cnt + 1;
                        else
                            state <= S_CONV3_bias;
                        end if;
                    when S_LINEAR_bias=> -- load bias
                        if (out_ch_cnt=16 and mem_read_cnt=2) then
                            state <= S_LINEAR_input;
                        end if;
                    when S_LINEAR_input =>  -- load input
                        if (mem_read_cnt=2) then
                            state <= S_LINEAR_wgt;
                        end if;
                    when S_LINEAR_wgt =>    -- load weight
                        if (mem_read_cnt=2) then
                            state <= S_LINEAR_EX;
                        end if;
                    when S_LINEAR_EX => -- execution state
                        -- have 32 output neuron, loop through 32 output neuron
                        if (out_ch_cnt=31) then
                            -- input neuron will be 8 or 32
                            -- 8: only need to calculate 1 time
                            --32: since memory only read 16 data, need 2 loop
                            if (in_channel=8 or in_ch_cnt(4)='1') then
                                state <= S_LINEAR_out;
                            else
                                state <= S_LINEAR_input;
                            end if;
                        else
                            state <= S_LINEAR_wgt;
                        end if;
                    when S_LINEAR_out => -- output state
                        -- have 32 output neuron, need 2 clock (each clock 16 data)
                        -- out_ch_cnt: first clock=0, second clock=16
                        if(out_ch_cnt=16) then
                            state <= S_LOAD_INFO;
                            info_cnt <= info_cnt+1;
                        end if;
                    when S_SWAP_input => -- load input
                        -- 2 clock delay from memory, 16 clock to read(256 data), total 18 clock
                        if (in_ch_cnt=17) then
                            state <= S_SWAP_out;
                        end if;
                    when S_SWAP_out => -- write back to memory
                        -- takes 16 clock to write(256 data, 16 data each clock)
                        if (out_ch_cnt=15) then
                            state <= S_LOAD_INFO;
                            info_cnt <= info_cnt+1;
                        end if;
                    when S_CONVT_bias => -- load bias
                        if (mem_read_cnt=2) then
                            state <= S_CONVT_wgt;
                        end if;
                    when S_CONVT_wgt => -- load weight
                        if (mem_read_cnt=2) then
                            state <= S_CONVT_input;
                        end if;
                    when S_CONVT_input =>   -- load input
                        if (mem_read_cnt=2) then
                            state <= S_CONVT_ex;
                        end if;
                    when S_CONVT_ex =>  -- execution state
                        if (in_size(5)='1' and rowcol_cnt(0)='0') then
                            -- same channel, same row need 2 time when input size=32
                            state <= S_CONVT_input;
                        else
                            if (in_ch_cnt=in_channel-1) then
                                -- finish this row for all input channel
                                -- prepare to output
                                state <= S_CONVT_addbias;
                            else
                                -- go to next input channel and keep accumulate
                                state <= S_CONVT_wgt;
                            end if;
                        end if;
                    when S_CONVT_addbias => 
                        -- add bias to output part of input channel accumulator
                        state <= S_CONVT_out;
                    when S_CONVT_out => -- output state
                        -- output 2 row of output image
                        -- can output 16 data each clock (16x16bit data width to memory)
                        -- input size= 4, output size= 8, 1 clock each row => 2 clock for 2 row
                        -- input size= 8, output size=16, 1 clock each row => 2 clock for 2 row
                        -- input size=16, output size=32, 2 clock each row => 4 clock for 2 row
                        -- input size=32, output size=64, 4 clock each row => 8 clock for 2 row
                        if ((in_size(2)='1' and out_rowcol_cnt(0)='0') or 
                            (in_size(3)='1' and out_rowcol_cnt(0)='0') or 
                            (in_size(4)='1' and out_rowcol_cnt(1 downto 0)="01") or
                            (in_size(5)='1' and out_rowcol_cnt(2 downto 0)="011")) then

                            -- ex: when in_size=4 and calculate 4 row => next output channel
                            -- ex: when in_size=32 and calculate 64 time (32 row, each row 2 time) => next output channel
                            -- otherwise, go to calculate next row of same output channel
                            if ((in_size(2)='1' and rowcol_cnt(1 downto 0)=3) or
                                (in_size(3)='1' and rowcol_cnt(2 downto 0)=7) or
                                (in_size(4)='1' and rowcol_cnt(3 downto 0)=15) or
                                (in_size(5)='1' and rowcol_cnt(5 downto 0)=63)) then
                                if (out_ch_cnt=out_channel-1) then
                                    -- finish all the output channel, finish this layer
                                    state <= S_CONVT_endbias;
                                else 
                                    -- next output channel
                                    state <= S_CONVT_bias;
                                end if;
                            else
                               state <=  S_CONVT_wgt;
                            end if;
                        end if;
                    when S_CONVT_endbias =>
                        -- ready to output the last row of this layer
                        -- add bias to output part of input channel accumulator
                        state <= S_CONVT_endout;
                    when S_CONVT_endout =>
                        -- ready to output the last row of this layer
                        -- in_size=4 or 8 => out_size=8 or 16 => output 1 clock
                        -- in_size=16 => out_size=32 => output 2 clock
                        -- in_size=32 => out_size=64 => output 4 clock
                        if (in_size(2)='1' or in_size(3)='1' or
                            (in_size(4)='1' and out_rowcol_cnt(0)='1') or
                            (in_size(5)='1' and out_rowcol_cnt(1 downto 0)="11")) then
                            state <= S_LOAD_INFO;
                            info_cnt <= info_cnt + 1;
                        end if;
                    when S_END => -- IDLE state
                        info_cnt <= (others => '0');
                        if (start='1') then
                            state <= S_LOAD_INFO;
                        end if;
                    when others =>
                        state <= S_END;
                end case;
            end if;
        end if;
    end process;
    ----------------------------------------------------------------------------

    -- layer info --------------------------------------------------------------
    load_layer_info <= '1' when(state=S_LOAD_INFO) else '0';
    layer_info_nxt <= mem_in(112 downto 0) when(load_layer_info='1') else layer_info;

    -- decoding layer information
    sigmoid_en <= layer_info(112);
    layer_type <= layer_info(111 downto 108);
    in_size <= unsigned(layer_info(107 downto 99));
    in_channel <= unsigned(layer_info(98 downto 90));
    out_channel <= unsigned(layer_info(89 downto 81));
    relu_en <= layer_info(80);
    wgt_addr <= unsigned(layer_info(79 downto 64));
    bias_addr <= unsigned(layer_info(63 downto 48));
    in_addr <= unsigned(layer_info(47 downto 32));
    in_wgt_addr2 <= unsigned(layer_info(31 downto 16));
    out_addr <= unsigned(layer_info(15 downto 0));
    

    process(clock)
    begin
        if (rising_edge(clock)) then
            if (reset='0') then
                layer_info <= (others => '0');
            else
                layer_info <= layer_info_nxt;
            end if;
        end if;
    end process;
    ----------------------------------------------------------------------------

    -- execution ------------------------------------------------------------------

    -- if sigmoid_en='1', output go through sigmoid function
    -- if relu_en='1', output go through relu function
    -- if overflow, set to max or min value
    memout0: for i in 0 to 15 generate
        mem_out(i*16+15 downto i*16) <= 
                 std_logic_vector(sigmoid_out(i))               when(sigmoid_en='1')
            else (others => '0')                                when(relu_en='1' and in_ch_accu(i)(31)='1')
            else std_logic_vector(in_ch_accu(i)(20 downto 5))   when(in_ch_accu(i)(31 downto 20)=x"FFF" or in_ch_accu(i)(31 downto 20)=x"000")
            else "0111111111111111"                             when(in_ch_accu(i)(31)='0')
            else "1000000000000000";
    end generate;

    -- sigmoid function
    sigmoid_gen : for i in 0 to 15 generate
        sigmoid0: sigmoid port map (
            in0 => in_ch_accu(i), out0 => sigmoid_out(i)
        );
    end generate;

    process(input_buf, wgt_buf, in_ch_cnt, out_ch_cnt, state, in_addr,
            in_wgt_addr2, info_cnt, mem_in, button, in_size, wgt_addr, 
            layer_info, in_ch_accu, bias_addr, mac_out, out_addr, mem_read_cnt,
            shf_reg1, shf_reg2, in_channel, out_channel, muladd_mul, ch_cnt,
            out_ch_cnt_mod16, ch_cnt_mod16, rowcol_cnt, mac_mul, out_rowcol_cnt,
            mem_in_sel16_out_ch, bias_buf1, bias_buf2)
    begin

        -- default
        input_buf_nxt <= input_buf;
        wgt_buf_nxt <= wgt_buf;
        bias_buf2_nxt <= bias_buf2;
        bias_buf1_nxt <= bias_buf1;
        in_ch_cnt_nxt <= in_ch_cnt;
        out_ch_cnt_nxt <= out_ch_cnt;
        ch_cnt_nxt <= ch_cnt;
        in_ch_accu_nxt <= in_ch_accu;
        rowcol_cnt_nxt <= rowcol_cnt;
        out_rowcol_cnt_nxt <= out_rowcol_cnt;
        shf_reg1_nxt <= shf_reg1;
        shf_reg2_nxt <= shf_reg2;
        mem_read_cnt_nxt <= (others => '0');

        mem_addr_0 <= (others => '0');
        mem_en <= '0';
        mem_RW <= read;


        case (state) is
            when S_LOAD_INFO => -- load layer information from memory
                -- set memory address
                mem_addr_0 <= resize(info_cnt,16);
                mem_en <= '1';
                mem_RW <= read;

                in_ch_cnt_nxt <= (others => '0');
                out_ch_cnt_nxt <= (others => '0');
                ch_cnt_nxt <= (others => '0');
                rowcol_cnt_nxt <= (others => '0');
                out_rowcol_cnt_nxt <= (others => '1');
                in_ch_accu_nxt <= (others => (others => '0'));
                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_INFO =>  -- decode layer information
                in_ch_cnt_nxt <= (others => '0');
                out_ch_cnt_nxt <= (others => '0');
                ch_cnt_nxt <= (others => '0');
                rowcol_cnt_nxt <= (others => '0');
                out_rowcol_cnt_nxt <= (others => '1');
                in_ch_accu_nxt <= (others => (others => '0'));
            when S_LINEAR_bias => -- load linear bias
                -- set address
                -- 16 bias store in a row
                mem_addr_0 <= bias_addr + out_ch_cnt(8 downto 4);
                mem_en <= '1';
                mem_RW <= read;

                -- load to input channel accumulator (index 0~31)
                if (mem_read_cnt=2) then -- delay 2 clocks for memory read
                    for i in 0 to 15 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i+16);
                    end loop;
                    for i in 0 to 15 loop
                        in_ch_accu_nxt(i+16) <= resize(signed(mem_in(i*16+15 downto i*16)&"00000"),32);
                    end loop;
                    if (out_ch_cnt=0) then
                        out_ch_cnt_nxt <= to_unsigned(16,9);
                    else
                        out_ch_cnt_nxt <= to_unsigned(0,9);
                    end if;
                end if;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_LINEAR_input =>  -- load input
                -- set address
                -- 16 input neuron stored in a row
                mem_addr_0 <= in_addr + in_ch_cnt(8 downto 4);
                mem_en <= '1';
                mem_RW <= read;

                -- load to input buffer
                -- when input_neuron=8 => input come from button
                -- when input_neuron=32 => input come from memory
                if (in_channel(3)='1') then
                    for i in 0 to 7 loop
                        input_buf_nxt(i) <= signed(button(i*16+15 downto i*16));
                    end loop;
                    for i in 0 to 7 loop
                        input_buf_nxt(i+8) <= signed(button(i*16+15 downto i*16));
                    end loop;
                else --in_channel=32
                    for i in 0 to 15 loop
                        input_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                    end loop;
                end if;

                out_ch_cnt_nxt <= to_unsigned(0,9);

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_LINEAR_wgt => -- load weight
                mem_en <= '1';
                mem_RW <= read;

                -- if input channel=8 , 2 channel stored in 1 row (each row 16 data = 8x2)
                -- if input channel=32, 1 channel stored in 2 row
                if (in_channel=8) then
                    mem_addr_0 <= wgt_addr + out_ch_cnt(8 downto 1);
                    if (out_ch_cnt(0)='0') then
                        for i in 0 to 7 loop
                            wgt_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                        end loop;
                        for i in 8 to 15 loop
                            wgt_buf_nxt(i) <= to_signed(0,16);
                        end loop;
                    else
                        for i in 0 to 7 loop
                            wgt_buf_nxt(i) <= to_signed(0,16);
                        end loop;
                        for i in 8 to 15 loop
                            wgt_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                        end loop;
                    end if;
                else -- input channel = 32
                    mem_addr_0 <= wgt_addr+(out_ch_cnt&in_ch_cnt(4));
                    for i in 0 to 15 loop
                        wgt_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                    end loop;
                end if;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_LINEAR_EX =>
                -- have only 16 mac module, need 2 times to calculate 32 output neuron
                for i in 0 to 15 loop
                    if (out_ch_cnt(4 downto 0)=i) then
                        in_ch_accu_nxt(i) <= in_ch_accu(i)+mac_out(i);
                    end if;
                end loop;
                for i in 16 to 31 loop
                    if (out_ch_cnt(4 downto 0)=i) then
                        in_ch_accu_nxt(i) <= in_ch_accu(i)+mac_out(i-16);
                    end if;
                end loop;

                out_ch_cnt_nxt <= out_ch_cnt+1;
                if (out_ch_cnt=31) then
                    in_ch_cnt_nxt <= in_ch_cnt+16;
                    out_ch_cnt_nxt <= to_unsigned(0,9);
                end if;
            when S_LINEAR_out => -- output state
                if (out_ch_cnt=0) then
                    mem_addr_0 <= out_addr;
                    out_ch_cnt_nxt <= to_unsigned(16,9);
                else
                    mem_addr_0 <= out_addr+1;
                    out_ch_cnt_nxt <= to_unsigned(0,9);
                end if;
                mem_en <= '1';
                mem_RW <= write;

                -- shift data to output
                for i in 0 to 15 loop
                    in_ch_accu_nxt(i) <= in_ch_accu(i+16);
                end loop;
            when S_MULADD_mul => -- load multiplication input
                -- set address
                mem_addr_0 <= in_wgt_addr2;
                mem_en <= '1';
                mem_RW <= read;

                -- load to shift register
                for i in 0 to 15 loop
                    shf_reg1_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                end loop;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_MULADD_add => -- load addition input
                -- set address
                mem_addr_0 <= in_wgt_addr2(15 downto 1)&"1";
                mem_en <= '1';
                mem_RW <= read;

                -- load to shift register
                for i in 0 to 15 loop
                    shf_reg2_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                end loop;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_MULADD_input => -- load input
                -- set address
                mem_addr_0 <= in_addr + out_ch_cnt;
                mem_en <= '1';
                mem_RW <= read;

                -- load to input buffer
                for i in 0 to 15 loop
                    input_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                end loop;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_MULADD_ex => -- execution state
                -- output = input*mul+input+add
                for i in 0 to 15 loop
                    -- muladd_mul=input*mul
                    in_ch_accu_nxt(i) <= resize(muladd_mul(i)(31 downto 8),32)+
                                         (input_buf(i)&"00000")+(shf_reg2(0)&"00000");
                end loop;
            when S_MULADD_out => -- output state
                -- set address
                mem_addr_0 <= out_addr + out_ch_cnt;
                mem_en <= '1';
                mem_RW <= write;

                out_ch_cnt_nxt <= out_ch_cnt + 1;

                -- shift register to get next add & mul data
                for i in 0 to 14 loop
                    shf_reg1_nxt(i) <= shf_reg1(i+1);
                    shf_reg2_nxt(i) <= shf_reg2(i+1);
                end loop;
            when S_CONV3_bias => -- load bias
                -- set address (bias of 16 output channel stored in a row)
                mem_addr_0 <= bias_addr + out_ch_cnt(8 downto 4);
                mem_en <= '1';
                mem_RW <= read;

                -- load bias to input channel accumulator
                -- mem_in_sel16_out_ch is 16-1 multiplexer (choosing 1 from 16 bias read from memory)
                for i in 0 to 15 loop
                    in_ch_accu_nxt(i) <= mem_in_sel16_out_ch;
                end loop;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_CONV3_wgt1 => -- load first 8 weight (total 3x3=9 weight)
                -- set address
                if (mem_read_cnt=0) then -- address of first 8 weight
                    mem_addr_0 <= wgt_addr+ch_cnt(14 downto 1);
                elsif (mem_read_cnt=1) then -- address of last 1 weight
                    mem_addr_0 <= in_wgt_addr2+ch_cnt(14 downto 4);
                else -- address of input
                    mem_addr_0 <= in_addr+in_ch_cnt;
                end if;
                mem_en <= '1';
                mem_RW <= read;

                -- choosing upper 8 data or lower 8 data (8x2 store in each row)
                for i in 0 to 7 loop
                    if (ch_cnt(0)='0') then
                        wgt_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                    else
                        wgt_buf_nxt(i) <= signed(mem_in((i+8)*16+15 downto (i+8)*16));
                    end if;
                end loop;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_CONV3_wgt2 => -- load last 1 weight of 3x3=9 weight
                mem_en <= '1';
                mem_RW <= read;

                -- load to index 8 (9th) weight buffer
                -- 16-1 multiplexer
                wgt_buf_nxt(8) <= signed(mem_in(ch_cnt_mod16*16+15 downto ch_cnt_mod16*16));
                
                for i in 9 to 15 loop
                    wgt_buf_nxt(i) <= (others => '0');
                end loop;
            when S_CONV3_input => -- load 4x4 input
                mem_en <= '1';
                mem_RW <= read;

                -- 4x4=16 input pixel load to input buffer
                for i in 0 to 15 loop
                    input_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                end loop;
            when S_CONV3_ex => -- execution state

                -- accumulate mac output for all input channel
                for i in 0 to 15 loop
                    in_ch_accu_nxt(i) <= in_ch_accu(i) + mac_out(i);
                end loop;

                in_ch_cnt_nxt <= in_ch_cnt + 1;
                ch_cnt_nxt <= ch_cnt + 1;

            when S_CONV3_out => -- output state: output 1 output channel, size=4x4 => 1 clock
                -- set address
                mem_addr_0 <= out_addr + out_ch_cnt;
                mem_en <= '1';
                mem_RW <= write;

                out_ch_cnt_nxt <= out_ch_cnt + 1;
                in_ch_cnt_nxt <= (others => '0');

            when S_SWAP_input => -- replace oldest frame input by latest frame

                -- set address
                if (in_ch_cnt(3)='1') then -- read latest frame, move to 4th
                    mem_addr_0 <= in_addr + in_ch_cnt(1 downto 0);
                else -- read 2nd ~ 4th frame, move to 1st ~ 3rd
                    mem_addr_0 <= out_addr + in_ch_cnt + 4;
                end if;
                mem_en <= '1';
                mem_RW <= read;

                in_ch_cnt_nxt <= in_ch_cnt + 1;

                -- each frame have 4 channel, 4 frame total 16 channel, each channel 4x4 pixel, total 256 data
                if (mem_read_cnt=2) then
                    for i in 0 to 239 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i+16);
                    end loop;
                    for i in 0 to 15 loop
                        in_ch_accu_nxt(i+240) <= resize(signed(mem_in(i*16+15 downto i*16)&"00000"),32);
                    end loop;
                end if;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= mem_read_cnt;
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;

            when S_SWAP_out => -- output state
                -- set address
                mem_addr_0 <= out_addr + out_ch_cnt;
                mem_en <= '1';
                mem_RW <= write;

                out_ch_cnt_nxt <= out_ch_cnt + 1;

                -- shift register to output data (16 clocks)
                for i in 0 to 239 loop
                    in_ch_accu_nxt(i) <= in_ch_accu(i+16);
                end loop;
            when S_CONVT_bias => -- load bias
                -- set address (each row store bias of 16 output channel)
                mem_addr_0 <= bias_addr + out_ch_cnt(8 downto 4);
                mem_en <= '1';
                mem_RW <= read;

                -- load bias to bias buffer
                bias_buf1_nxt <= mem_in_sel16_out_ch;

                if (mem_read_cnt=0) then
                    bias_buf2_nxt <= bias_buf1;
                end if;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_CONVT_wgt => -- load 4x4 kernel weight for each channel
                -- set address
                mem_addr_0 <= wgt_addr + ch_cnt;
                mem_en <= '1';
                mem_RW <= read;

                -- load to weight buffer
                for i in 0 to 15 loop
                    wgt_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                end loop;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_CONVT_input => -- load input
                -- if input size=4, each row of memory store 4 row of pixel
                -- read 1 row of pixel each time
                if (in_size(2)='1') then -- in_size=4
                    mem_addr_0 <= in_addr + rowcol_cnt(11 downto 2) + in_ch_cnt;
                elsif (in_size(3)='1') then -- in_size=8
                    mem_addr_0 <= in_addr + rowcol_cnt + unsigned(in_ch_cnt&"000");
                elsif (in_size(4)='1') then -- in_size=16
                    mem_addr_0 <= in_addr + rowcol_cnt + unsigned(in_ch_cnt&"0000");
                else-- in_size=32
                    mem_addr_0 <= in_addr + rowcol_cnt + unsigned(in_ch_cnt&"000000");
                end if;
                mem_en <= '1';
                mem_RW <= read;

                if (in_size=4) then -- in_size=4 => 4 row of image stored in 1 line in memory
                    for i in 0 to 3 loop
                        case (rowcol_cnt(1 downto 0)) is
                            when "00" =>
                                input_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                            when "01" =>
                                input_buf_nxt(i) <= signed(mem_in(i*16+79 downto i*16+64));
                            when "10" =>
                                input_buf_nxt(i) <= signed(mem_in(i*16+143 downto i*16+128));
                            when "11" =>
                                input_buf_nxt(i) <= signed(mem_in(i*16+207 downto i*16+192));
                            when others =>
                                input_buf_nxt(i) <= (others => '0');
                        end case;
                    end loop;
                    for i in 4 to 15 loop
                        input_buf_nxt(i) <= (others => '0');
                    end loop;
                elsif (in_size=8) then
                    for i in 0 to 7 loop
                        input_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                    end loop;
                    for i in 8 to 15 loop
                        input_buf_nxt(i) <= (others => '0');
                    end loop;
                else
                    for i in 0 to 15 loop
                        input_buf_nxt(i) <= signed(mem_in(i*16+15 downto i*16));
                    end loop;
                end if;

                if (mem_read_cnt=2) then
                    mem_read_cnt_nxt <= (others => '0');
                else
                    mem_read_cnt_nxt <= mem_read_cnt+1;
                end if;
            when S_CONVT_ex => -- execution state
                -- calculation of transposed convolution (hard to comment)
                if (in_size(5)='1' and rowcol_cnt(0)='1') then 
                -- in_size=32, need to calculate right half of input channel accumulator
                -- ex: 32~63, 96~127, 160~191, 224~255
                
                    if (rowcol_cnt(5 downto 0)/=1) then -- skip first row
                    in_ch_accu_nxt( 31)<= in_ch_accu( 31)+mac_mul( 0)( 0);
                    in_ch_accu_nxt( 32)<= in_ch_accu( 32)+mac_mul( 0)( 1);
                    in_ch_accu_nxt( 33)<= in_ch_accu( 33)+mac_mul( 0)( 2)+mac_mul( 1)( 0);
                    in_ch_accu_nxt( 34)<= in_ch_accu( 34)+mac_mul( 0)( 3)+mac_mul( 1)( 1);
                    in_ch_accu_nxt( 35)<= in_ch_accu( 35)+mac_mul( 1)( 2)+mac_mul( 2)( 0);
                    in_ch_accu_nxt( 36)<= in_ch_accu( 36)+mac_mul( 1)( 3)+mac_mul( 2)( 1);
                    in_ch_accu_nxt( 37)<= in_ch_accu( 37)+mac_mul( 2)( 2)+mac_mul( 3)( 0);
                    in_ch_accu_nxt( 38)<= in_ch_accu( 38)+mac_mul( 2)( 3)+mac_mul( 3)( 1);
                    in_ch_accu_nxt( 39)<= in_ch_accu( 39)+mac_mul( 3)( 2)+mac_mul( 4)( 0);
                    in_ch_accu_nxt( 40)<= in_ch_accu( 40)+mac_mul( 3)( 3)+mac_mul( 4)( 1);
                    in_ch_accu_nxt( 41)<= in_ch_accu( 41)+mac_mul( 4)( 2)+mac_mul( 5)( 0);
                    in_ch_accu_nxt( 42)<= in_ch_accu( 42)+mac_mul( 4)( 3)+mac_mul( 5)( 1);
                    in_ch_accu_nxt( 43)<= in_ch_accu( 43)+mac_mul( 5)( 2)+mac_mul( 6)( 0);
                    in_ch_accu_nxt( 44)<= in_ch_accu( 44)+mac_mul( 5)( 3)+mac_mul( 6)( 1);
                    in_ch_accu_nxt( 45)<= in_ch_accu( 45)+mac_mul( 6)( 2)+mac_mul( 7)( 0);
                    in_ch_accu_nxt( 46)<= in_ch_accu( 46)+mac_mul( 6)( 3)+mac_mul( 7)( 1);
                    in_ch_accu_nxt( 47)<= in_ch_accu( 47)+mac_mul( 7)( 2)+mac_mul( 8)( 0);
                    in_ch_accu_nxt( 48)<= in_ch_accu( 48)+mac_mul( 7)( 3)+mac_mul( 8)( 1);
                    in_ch_accu_nxt( 49)<= in_ch_accu( 49)+mac_mul( 8)( 2)+mac_mul( 9)( 0);
                    in_ch_accu_nxt( 50)<= in_ch_accu( 50)+mac_mul( 8)( 3)+mac_mul( 9)( 1);
                    in_ch_accu_nxt( 51)<= in_ch_accu( 51)+mac_mul( 9)( 2)+mac_mul(10)( 0);
                    in_ch_accu_nxt( 52)<= in_ch_accu( 52)+mac_mul( 9)( 3)+mac_mul(10)( 1);
                    in_ch_accu_nxt( 53)<= in_ch_accu( 53)+mac_mul(10)( 2)+mac_mul(11)( 0);
                    in_ch_accu_nxt( 54)<= in_ch_accu( 54)+mac_mul(10)( 3)+mac_mul(11)( 1);
                    in_ch_accu_nxt( 55)<= in_ch_accu( 55)+mac_mul(11)( 2)+mac_mul(12)( 0);
                    in_ch_accu_nxt( 56)<= in_ch_accu( 56)+mac_mul(11)( 3)+mac_mul(12)( 1);
                    in_ch_accu_nxt( 57)<= in_ch_accu( 57)+mac_mul(12)( 2)+mac_mul(13)( 0);
                    in_ch_accu_nxt( 58)<= in_ch_accu( 58)+mac_mul(12)( 3)+mac_mul(13)( 1);
                    in_ch_accu_nxt( 59)<= in_ch_accu( 59)+mac_mul(13)( 2)+mac_mul(14)( 0);
                    in_ch_accu_nxt( 60)<= in_ch_accu( 60)+mac_mul(13)( 3)+mac_mul(14)( 1);
                    in_ch_accu_nxt( 61)<= in_ch_accu( 61)+mac_mul(14)( 2)+mac_mul(15)( 0);
                    in_ch_accu_nxt( 62)<= in_ch_accu( 62)+mac_mul(14)( 3)+mac_mul(15)( 1);
                    in_ch_accu_nxt( 63)<= in_ch_accu( 63)+mac_mul(15)( 2);
                    end if;

                    in_ch_accu_nxt( 95)<= in_ch_accu( 95)+mac_mul( 0)( 4);
                    in_ch_accu_nxt( 96)<= in_ch_accu( 96)+mac_mul( 0)( 5);
                    in_ch_accu_nxt( 97)<= in_ch_accu( 97)+mac_mul( 0)( 6)+mac_mul( 1)( 4);
                    in_ch_accu_nxt( 98)<= in_ch_accu( 98)+mac_mul( 0)( 7)+mac_mul( 1)( 5);
                    in_ch_accu_nxt( 99)<= in_ch_accu( 99)+mac_mul( 1)( 6)+mac_mul( 2)( 4);
                    in_ch_accu_nxt(100)<= in_ch_accu(100)+mac_mul( 1)( 7)+mac_mul( 2)( 5);
                    in_ch_accu_nxt(101)<= in_ch_accu(101)+mac_mul( 2)( 6)+mac_mul( 3)( 4);
                    in_ch_accu_nxt(102)<= in_ch_accu(102)+mac_mul( 2)( 7)+mac_mul( 3)( 5);
                    in_ch_accu_nxt(103)<= in_ch_accu(103)+mac_mul( 3)( 6)+mac_mul( 4)( 4);
                    in_ch_accu_nxt(104)<= in_ch_accu(104)+mac_mul( 3)( 7)+mac_mul( 4)( 5);
                    in_ch_accu_nxt(105)<= in_ch_accu(105)+mac_mul( 4)( 6)+mac_mul( 5)( 4);
                    in_ch_accu_nxt(106)<= in_ch_accu(106)+mac_mul( 4)( 7)+mac_mul( 5)( 5);
                    in_ch_accu_nxt(107)<= in_ch_accu(107)+mac_mul( 5)( 6)+mac_mul( 6)( 4);
                    in_ch_accu_nxt(108)<= in_ch_accu(108)+mac_mul( 5)( 7)+mac_mul( 6)( 5);
                    in_ch_accu_nxt(109)<= in_ch_accu(109)+mac_mul( 6)( 6)+mac_mul( 7)( 4);
                    in_ch_accu_nxt(110)<= in_ch_accu(110)+mac_mul( 6)( 7)+mac_mul( 7)( 5);
                    in_ch_accu_nxt(111)<= in_ch_accu(111)+mac_mul( 7)( 6)+mac_mul( 8)( 4);
                    in_ch_accu_nxt(112)<= in_ch_accu(112)+mac_mul( 7)( 7)+mac_mul( 8)( 5);
                    in_ch_accu_nxt(113)<= in_ch_accu(113)+mac_mul( 8)( 6)+mac_mul( 9)( 4);
                    in_ch_accu_nxt(114)<= in_ch_accu(114)+mac_mul( 8)( 7)+mac_mul( 9)( 5);
                    in_ch_accu_nxt(115)<= in_ch_accu(115)+mac_mul( 9)( 6)+mac_mul(10)( 4);
                    in_ch_accu_nxt(116)<= in_ch_accu(116)+mac_mul( 9)( 7)+mac_mul(10)( 5);
                    in_ch_accu_nxt(117)<= in_ch_accu(117)+mac_mul(10)( 6)+mac_mul(11)( 4);
                    in_ch_accu_nxt(118)<= in_ch_accu(118)+mac_mul(10)( 7)+mac_mul(11)( 5);
                    in_ch_accu_nxt(119)<= in_ch_accu(119)+mac_mul(11)( 6)+mac_mul(12)( 4);
                    in_ch_accu_nxt(120)<= in_ch_accu(120)+mac_mul(11)( 7)+mac_mul(12)( 5);
                    in_ch_accu_nxt(121)<= in_ch_accu(121)+mac_mul(12)( 6)+mac_mul(13)( 4);
                    in_ch_accu_nxt(122)<= in_ch_accu(122)+mac_mul(12)( 7)+mac_mul(13)( 5);
                    in_ch_accu_nxt(123)<= in_ch_accu(123)+mac_mul(13)( 6)+mac_mul(14)( 4);
                    in_ch_accu_nxt(124)<= in_ch_accu(124)+mac_mul(13)( 7)+mac_mul(14)( 5);
                    in_ch_accu_nxt(125)<= in_ch_accu(125)+mac_mul(14)( 6)+mac_mul(15)( 4);
                    in_ch_accu_nxt(126)<= in_ch_accu(126)+mac_mul(14)( 7)+mac_mul(15)( 5);
                    in_ch_accu_nxt(127)<= in_ch_accu(127)+mac_mul(15)( 6);

                    in_ch_accu_nxt(159)<= in_ch_accu(159)+mac_mul( 0)( 8);
                    in_ch_accu_nxt(160)<= in_ch_accu(160)+mac_mul( 0)( 9);
                    in_ch_accu_nxt(161)<= in_ch_accu(161)+mac_mul( 0)(10)+mac_mul( 1)( 8);
                    in_ch_accu_nxt(162)<= in_ch_accu(162)+mac_mul( 0)(11)+mac_mul( 1)( 9);
                    in_ch_accu_nxt(163)<= in_ch_accu(163)+mac_mul( 1)(10)+mac_mul( 2)( 8);
                    in_ch_accu_nxt(164)<= in_ch_accu(164)+mac_mul( 1)(11)+mac_mul( 2)( 9);
                    in_ch_accu_nxt(165)<= in_ch_accu(165)+mac_mul( 2)(10)+mac_mul( 3)( 8);
                    in_ch_accu_nxt(166)<= in_ch_accu(166)+mac_mul( 2)(11)+mac_mul( 3)( 9);
                    in_ch_accu_nxt(167)<= in_ch_accu(167)+mac_mul( 3)(10)+mac_mul( 4)( 8);
                    in_ch_accu_nxt(168)<= in_ch_accu(168)+mac_mul( 3)(11)+mac_mul( 4)( 9);
                    in_ch_accu_nxt(169)<= in_ch_accu(169)+mac_mul( 4)(10)+mac_mul( 5)( 8);
                    in_ch_accu_nxt(170)<= in_ch_accu(170)+mac_mul( 4)(11)+mac_mul( 5)( 9);
                    in_ch_accu_nxt(171)<= in_ch_accu(171)+mac_mul( 5)(10)+mac_mul( 6)( 8);
                    in_ch_accu_nxt(172)<= in_ch_accu(172)+mac_mul( 5)(11)+mac_mul( 6)( 9);
                    in_ch_accu_nxt(173)<= in_ch_accu(173)+mac_mul( 6)(10)+mac_mul( 7)( 8);
                    in_ch_accu_nxt(174)<= in_ch_accu(174)+mac_mul( 6)(11)+mac_mul( 7)( 9);
                    in_ch_accu_nxt(175)<= in_ch_accu(175)+mac_mul( 7)(10)+mac_mul( 8)( 8);
                    in_ch_accu_nxt(176)<= in_ch_accu(176)+mac_mul( 7)(11)+mac_mul( 8)( 9);
                    in_ch_accu_nxt(177)<= in_ch_accu(177)+mac_mul( 8)(10)+mac_mul( 9)( 8);
                    in_ch_accu_nxt(178)<= in_ch_accu(178)+mac_mul( 8)(11)+mac_mul( 9)( 9);
                    in_ch_accu_nxt(179)<= in_ch_accu(179)+mac_mul( 9)(10)+mac_mul(10)( 8);
                    in_ch_accu_nxt(180)<= in_ch_accu(180)+mac_mul( 9)(11)+mac_mul(10)( 9);
                    in_ch_accu_nxt(181)<= in_ch_accu(181)+mac_mul(10)(10)+mac_mul(11)( 8);
                    in_ch_accu_nxt(182)<= in_ch_accu(182)+mac_mul(10)(11)+mac_mul(11)( 9);
                    in_ch_accu_nxt(183)<= in_ch_accu(183)+mac_mul(11)(10)+mac_mul(12)( 8);
                    in_ch_accu_nxt(184)<= in_ch_accu(184)+mac_mul(11)(11)+mac_mul(12)( 9);
                    in_ch_accu_nxt(185)<= in_ch_accu(185)+mac_mul(12)(10)+mac_mul(13)( 8);
                    in_ch_accu_nxt(186)<= in_ch_accu(186)+mac_mul(12)(11)+mac_mul(13)( 9);
                    in_ch_accu_nxt(187)<= in_ch_accu(187)+mac_mul(13)(10)+mac_mul(14)( 8);
                    in_ch_accu_nxt(188)<= in_ch_accu(188)+mac_mul(13)(11)+mac_mul(14)( 9);
                    in_ch_accu_nxt(189)<= in_ch_accu(189)+mac_mul(14)(10)+mac_mul(15)( 8);
                    in_ch_accu_nxt(190)<= in_ch_accu(190)+mac_mul(14)(11)+mac_mul(15)( 9);
                    in_ch_accu_nxt(191)<= in_ch_accu(191)+mac_mul(15)(10);

                    if (rowcol_cnt(5 downto 0)/=63) then -- skip last row
                    in_ch_accu_nxt(223)<= in_ch_accu(223)+mac_mul( 0)(12);
                    in_ch_accu_nxt(224)<= in_ch_accu(224)+mac_mul( 0)(13);
                    in_ch_accu_nxt(225)<= in_ch_accu(225)+mac_mul( 0)(14)+mac_mul( 1)(12);
                    in_ch_accu_nxt(226)<= in_ch_accu(226)+mac_mul( 0)(15)+mac_mul( 1)(13);
                    in_ch_accu_nxt(227)<= in_ch_accu(227)+mac_mul( 1)(14)+mac_mul( 2)(12);
                    in_ch_accu_nxt(228)<= in_ch_accu(228)+mac_mul( 1)(15)+mac_mul( 2)(13);
                    in_ch_accu_nxt(229)<= in_ch_accu(229)+mac_mul( 2)(14)+mac_mul( 3)(12);
                    in_ch_accu_nxt(230)<= in_ch_accu(230)+mac_mul( 2)(15)+mac_mul( 3)(13);
                    in_ch_accu_nxt(231)<= in_ch_accu(231)+mac_mul( 3)(14)+mac_mul( 4)(12);
                    in_ch_accu_nxt(232)<= in_ch_accu(232)+mac_mul( 3)(15)+mac_mul( 4)(13);
                    in_ch_accu_nxt(233)<= in_ch_accu(233)+mac_mul( 4)(14)+mac_mul( 5)(12);
                    in_ch_accu_nxt(234)<= in_ch_accu(234)+mac_mul( 4)(15)+mac_mul( 5)(13);
                    in_ch_accu_nxt(235)<= in_ch_accu(235)+mac_mul( 5)(14)+mac_mul( 6)(12);
                    in_ch_accu_nxt(236)<= in_ch_accu(236)+mac_mul( 5)(15)+mac_mul( 6)(13);
                    in_ch_accu_nxt(237)<= in_ch_accu(237)+mac_mul( 6)(14)+mac_mul( 7)(12);
                    in_ch_accu_nxt(238)<= in_ch_accu(238)+mac_mul( 6)(15)+mac_mul( 7)(13);
                    in_ch_accu_nxt(239)<= in_ch_accu(239)+mac_mul( 7)(14)+mac_mul( 8)(12);
                    in_ch_accu_nxt(240)<= in_ch_accu(240)+mac_mul( 7)(15)+mac_mul( 8)(13);
                    in_ch_accu_nxt(241)<= in_ch_accu(241)+mac_mul( 8)(14)+mac_mul( 9)(12);
                    in_ch_accu_nxt(242)<= in_ch_accu(242)+mac_mul( 8)(15)+mac_mul( 9)(13);
                    in_ch_accu_nxt(243)<= in_ch_accu(243)+mac_mul( 9)(14)+mac_mul(10)(12);
                    in_ch_accu_nxt(244)<= in_ch_accu(244)+mac_mul( 9)(15)+mac_mul(10)(13);
                    in_ch_accu_nxt(245)<= in_ch_accu(245)+mac_mul(10)(14)+mac_mul(11)(12);
                    in_ch_accu_nxt(246)<= in_ch_accu(246)+mac_mul(10)(15)+mac_mul(11)(13);
                    in_ch_accu_nxt(247)<= in_ch_accu(247)+mac_mul(11)(14)+mac_mul(12)(12);
                    in_ch_accu_nxt(248)<= in_ch_accu(248)+mac_mul(11)(15)+mac_mul(12)(13);
                    in_ch_accu_nxt(249)<= in_ch_accu(249)+mac_mul(12)(14)+mac_mul(13)(12);
                    in_ch_accu_nxt(250)<= in_ch_accu(250)+mac_mul(12)(15)+mac_mul(13)(13);
                    in_ch_accu_nxt(251)<= in_ch_accu(251)+mac_mul(13)(14)+mac_mul(14)(12);
                    in_ch_accu_nxt(252)<= in_ch_accu(252)+mac_mul(13)(15)+mac_mul(14)(13);
                    in_ch_accu_nxt(253)<= in_ch_accu(253)+mac_mul(14)(14)+mac_mul(15)(12);
                    in_ch_accu_nxt(254)<= in_ch_accu(254)+mac_mul(14)(15)+mac_mul(15)(13);
                    in_ch_accu_nxt(255)<= in_ch_accu(255)+mac_mul(15)(14);
                    end if;

                else

                    if (rowcol_cnt(5 downto 0)/=0) then -- skip first row
                    in_ch_accu_nxt(  0)<= in_ch_accu(  0)+mac_mul( 0)( 1);
                    in_ch_accu_nxt(  1)<= in_ch_accu(  1)+mac_mul( 0)( 2)+mac_mul( 1)( 0);
                    in_ch_accu_nxt(  2)<= in_ch_accu(  2)+mac_mul( 0)( 3)+mac_mul( 1)( 1);
                    in_ch_accu_nxt(  3)<= in_ch_accu(  3)+mac_mul( 1)( 2)+mac_mul( 2)( 0);
                    in_ch_accu_nxt(  4)<= in_ch_accu(  4)+mac_mul( 1)( 3)+mac_mul( 2)( 1);
                    in_ch_accu_nxt(  5)<= in_ch_accu(  5)+mac_mul( 2)( 2)+mac_mul( 3)( 0);
                    in_ch_accu_nxt(  6)<= in_ch_accu(  6)+mac_mul( 2)( 3)+mac_mul( 3)( 1);
                    in_ch_accu_nxt(  7)<= in_ch_accu(  7)+mac_mul( 3)( 2)+mac_mul( 4)( 0);
                    in_ch_accu_nxt(  8)<= in_ch_accu(  8)+mac_mul( 3)( 3)+mac_mul( 4)( 1);
                    in_ch_accu_nxt(  9)<= in_ch_accu(  9)+mac_mul( 4)( 2)+mac_mul( 5)( 0);
                    in_ch_accu_nxt( 10)<= in_ch_accu( 10)+mac_mul( 4)( 3)+mac_mul( 5)( 1);
                    in_ch_accu_nxt( 11)<= in_ch_accu( 11)+mac_mul( 5)( 2)+mac_mul( 6)( 0);
                    in_ch_accu_nxt( 12)<= in_ch_accu( 12)+mac_mul( 5)( 3)+mac_mul( 6)( 1);
                    in_ch_accu_nxt( 13)<= in_ch_accu( 13)+mac_mul( 6)( 2)+mac_mul( 7)( 0);
                    in_ch_accu_nxt( 14)<= in_ch_accu( 14)+mac_mul( 6)( 3)+mac_mul( 7)( 1);
                    in_ch_accu_nxt( 15)<= in_ch_accu( 15)+mac_mul( 7)( 2)+mac_mul( 8)( 0);
                    in_ch_accu_nxt( 16)<= in_ch_accu( 16)+mac_mul( 7)( 3)+mac_mul( 8)( 1);
                    in_ch_accu_nxt( 17)<= in_ch_accu( 17)+mac_mul( 8)( 2)+mac_mul( 9)( 0);
                    in_ch_accu_nxt( 18)<= in_ch_accu( 18)+mac_mul( 8)( 3)+mac_mul( 9)( 1);
                    in_ch_accu_nxt( 19)<= in_ch_accu( 19)+mac_mul( 9)( 2)+mac_mul(10)( 0);
                    in_ch_accu_nxt( 20)<= in_ch_accu( 20)+mac_mul( 9)( 3)+mac_mul(10)( 1);
                    in_ch_accu_nxt( 21)<= in_ch_accu( 21)+mac_mul(10)( 2)+mac_mul(11)( 0);
                    in_ch_accu_nxt( 22)<= in_ch_accu( 22)+mac_mul(10)( 3)+mac_mul(11)( 1);
                    in_ch_accu_nxt( 23)<= in_ch_accu( 23)+mac_mul(11)( 2)+mac_mul(12)( 0);
                    in_ch_accu_nxt( 24)<= in_ch_accu( 24)+mac_mul(11)( 3)+mac_mul(12)( 1);
                    in_ch_accu_nxt( 25)<= in_ch_accu( 25)+mac_mul(12)( 2)+mac_mul(13)( 0);
                    in_ch_accu_nxt( 26)<= in_ch_accu( 26)+mac_mul(12)( 3)+mac_mul(13)( 1);
                    in_ch_accu_nxt( 27)<= in_ch_accu( 27)+mac_mul(13)( 2)+mac_mul(14)( 0);
                    in_ch_accu_nxt( 28)<= in_ch_accu( 28)+mac_mul(13)( 3)+mac_mul(14)( 1);
                    in_ch_accu_nxt( 29)<= in_ch_accu( 29)+mac_mul(14)( 2)+mac_mul(15)( 0);
                    in_ch_accu_nxt( 30)<= in_ch_accu( 30)+mac_mul(14)( 3)+mac_mul(15)( 1);
                    in_ch_accu_nxt( 31)<= in_ch_accu( 31)+mac_mul(15)( 2);
                    in_ch_accu_nxt( 32)<= in_ch_accu( 32)+mac_mul(15)( 3);
                    end if;

                    in_ch_accu_nxt( 64)<= in_ch_accu( 64)+mac_mul( 0)( 5);
                    in_ch_accu_nxt( 65)<= in_ch_accu( 65)+mac_mul( 0)( 6)+mac_mul( 1)( 4);
                    in_ch_accu_nxt( 66)<= in_ch_accu( 66)+mac_mul( 0)( 7)+mac_mul( 1)( 5);
                    in_ch_accu_nxt( 67)<= in_ch_accu( 67)+mac_mul( 1)( 6)+mac_mul( 2)( 4);
                    in_ch_accu_nxt( 68)<= in_ch_accu( 68)+mac_mul( 1)( 7)+mac_mul( 2)( 5);
                    in_ch_accu_nxt( 69)<= in_ch_accu( 69)+mac_mul( 2)( 6)+mac_mul( 3)( 4);
                    in_ch_accu_nxt( 70)<= in_ch_accu( 70)+mac_mul( 2)( 7)+mac_mul( 3)( 5);
                    in_ch_accu_nxt( 71)<= in_ch_accu( 71)+mac_mul( 3)( 6)+mac_mul( 4)( 4);
                    in_ch_accu_nxt( 72)<= in_ch_accu( 72)+mac_mul( 3)( 7)+mac_mul( 4)( 5);
                    in_ch_accu_nxt( 73)<= in_ch_accu( 73)+mac_mul( 4)( 6)+mac_mul( 5)( 4);
                    in_ch_accu_nxt( 74)<= in_ch_accu( 74)+mac_mul( 4)( 7)+mac_mul( 5)( 5);
                    in_ch_accu_nxt( 75)<= in_ch_accu( 75)+mac_mul( 5)( 6)+mac_mul( 6)( 4);
                    in_ch_accu_nxt( 76)<= in_ch_accu( 76)+mac_mul( 5)( 7)+mac_mul( 6)( 5);
                    in_ch_accu_nxt( 77)<= in_ch_accu( 77)+mac_mul( 6)( 6)+mac_mul( 7)( 4);
                    in_ch_accu_nxt( 78)<= in_ch_accu( 78)+mac_mul( 6)( 7)+mac_mul( 7)( 5);
                    in_ch_accu_nxt( 79)<= in_ch_accu( 79)+mac_mul( 7)( 6)+mac_mul( 8)( 4);
                    in_ch_accu_nxt( 80)<= in_ch_accu( 80)+mac_mul( 7)( 7)+mac_mul( 8)( 5);
                    in_ch_accu_nxt( 81)<= in_ch_accu( 81)+mac_mul( 8)( 6)+mac_mul( 9)( 4);
                    in_ch_accu_nxt( 82)<= in_ch_accu( 82)+mac_mul( 8)( 7)+mac_mul( 9)( 5);
                    in_ch_accu_nxt( 83)<= in_ch_accu( 83)+mac_mul( 9)( 6)+mac_mul(10)( 4);
                    in_ch_accu_nxt( 84)<= in_ch_accu( 84)+mac_mul( 9)( 7)+mac_mul(10)( 5);
                    in_ch_accu_nxt( 85)<= in_ch_accu( 85)+mac_mul(10)( 6)+mac_mul(11)( 4);
                    in_ch_accu_nxt( 86)<= in_ch_accu( 86)+mac_mul(10)( 7)+mac_mul(11)( 5);
                    in_ch_accu_nxt( 87)<= in_ch_accu( 87)+mac_mul(11)( 6)+mac_mul(12)( 4);
                    in_ch_accu_nxt( 88)<= in_ch_accu( 88)+mac_mul(11)( 7)+mac_mul(12)( 5);
                    in_ch_accu_nxt( 89)<= in_ch_accu( 89)+mac_mul(12)( 6)+mac_mul(13)( 4);
                    in_ch_accu_nxt( 90)<= in_ch_accu( 90)+mac_mul(12)( 7)+mac_mul(13)( 5);
                    in_ch_accu_nxt( 91)<= in_ch_accu( 91)+mac_mul(13)( 6)+mac_mul(14)( 4);
                    in_ch_accu_nxt( 92)<= in_ch_accu( 92)+mac_mul(13)( 7)+mac_mul(14)( 5);
                    in_ch_accu_nxt( 93)<= in_ch_accu( 93)+mac_mul(14)( 6)+mac_mul(15)( 4);
                    in_ch_accu_nxt( 94)<= in_ch_accu( 94)+mac_mul(14)( 7)+mac_mul(15)( 5);
                    in_ch_accu_nxt( 95)<= in_ch_accu( 95)+mac_mul(15)( 6);
                    in_ch_accu_nxt( 96)<= in_ch_accu( 96)+mac_mul(15)( 7);

                    in_ch_accu_nxt(128)<= in_ch_accu(128)+mac_mul( 0)( 9);
                    in_ch_accu_nxt(129)<= in_ch_accu(129)+mac_mul( 0)(10)+mac_mul( 1)( 8);
                    in_ch_accu_nxt(130)<= in_ch_accu(130)+mac_mul( 0)(11)+mac_mul( 1)( 9);
                    in_ch_accu_nxt(131)<= in_ch_accu(131)+mac_mul( 1)(10)+mac_mul( 2)( 8);
                    in_ch_accu_nxt(132)<= in_ch_accu(132)+mac_mul( 1)(11)+mac_mul( 2)( 9);
                    in_ch_accu_nxt(133)<= in_ch_accu(133)+mac_mul( 2)(10)+mac_mul( 3)( 8);
                    in_ch_accu_nxt(134)<= in_ch_accu(134)+mac_mul( 2)(11)+mac_mul( 3)( 9);
                    in_ch_accu_nxt(135)<= in_ch_accu(135)+mac_mul( 3)(10)+mac_mul( 4)( 8);
                    in_ch_accu_nxt(136)<= in_ch_accu(136)+mac_mul( 3)(11)+mac_mul( 4)( 9);
                    in_ch_accu_nxt(137)<= in_ch_accu(137)+mac_mul( 4)(10)+mac_mul( 5)( 8);
                    in_ch_accu_nxt(138)<= in_ch_accu(138)+mac_mul( 4)(11)+mac_mul( 5)( 9);
                    in_ch_accu_nxt(139)<= in_ch_accu(139)+mac_mul( 5)(10)+mac_mul( 6)( 8);
                    in_ch_accu_nxt(140)<= in_ch_accu(140)+mac_mul( 5)(11)+mac_mul( 6)( 9);
                    in_ch_accu_nxt(141)<= in_ch_accu(141)+mac_mul( 6)(10)+mac_mul( 7)( 8);
                    in_ch_accu_nxt(142)<= in_ch_accu(142)+mac_mul( 6)(11)+mac_mul( 7)( 9);
                    in_ch_accu_nxt(143)<= in_ch_accu(143)+mac_mul( 7)(10)+mac_mul( 8)( 8);
                    in_ch_accu_nxt(144)<= in_ch_accu(144)+mac_mul( 7)(11)+mac_mul( 8)( 9);
                    in_ch_accu_nxt(145)<= in_ch_accu(145)+mac_mul( 8)(10)+mac_mul( 9)( 8);
                    in_ch_accu_nxt(146)<= in_ch_accu(146)+mac_mul( 8)(11)+mac_mul( 9)( 9);
                    in_ch_accu_nxt(147)<= in_ch_accu(147)+mac_mul( 9)(10)+mac_mul(10)( 8);
                    in_ch_accu_nxt(148)<= in_ch_accu(148)+mac_mul( 9)(11)+mac_mul(10)( 9);
                    in_ch_accu_nxt(149)<= in_ch_accu(149)+mac_mul(10)(10)+mac_mul(11)( 8);
                    in_ch_accu_nxt(150)<= in_ch_accu(150)+mac_mul(10)(11)+mac_mul(11)( 9);
                    in_ch_accu_nxt(151)<= in_ch_accu(151)+mac_mul(11)(10)+mac_mul(12)( 8);
                    in_ch_accu_nxt(152)<= in_ch_accu(152)+mac_mul(11)(11)+mac_mul(12)( 9);
                    in_ch_accu_nxt(153)<= in_ch_accu(153)+mac_mul(12)(10)+mac_mul(13)( 8);
                    in_ch_accu_nxt(154)<= in_ch_accu(154)+mac_mul(12)(11)+mac_mul(13)( 9);
                    in_ch_accu_nxt(155)<= in_ch_accu(155)+mac_mul(13)(10)+mac_mul(14)( 8);
                    in_ch_accu_nxt(156)<= in_ch_accu(156)+mac_mul(13)(11)+mac_mul(14)( 9);
                    in_ch_accu_nxt(157)<= in_ch_accu(157)+mac_mul(14)(10)+mac_mul(15)( 8);
                    in_ch_accu_nxt(158)<= in_ch_accu(158)+mac_mul(14)(11)+mac_mul(15)( 9);
                    in_ch_accu_nxt(159)<= in_ch_accu(159)+mac_mul(15)(10);
                    in_ch_accu_nxt(160)<= in_ch_accu(160)+mac_mul(15)(11);

                    if ((in_size(2)='1' and rowcol_cnt(1 downto 0)/=3) or
                        (in_size(3)='1' and rowcol_cnt(2 downto 0)/=7) or
                        (in_size(4)='1' and rowcol_cnt(3 downto 0)/=15) or
                        (in_size(5)='1' and rowcol_cnt(5 downto 0)/=62)) then -- skip last row
                    in_ch_accu_nxt(192)<= in_ch_accu(192)+mac_mul( 0)(13);
                    in_ch_accu_nxt(193)<= in_ch_accu(193)+mac_mul( 0)(14)+mac_mul( 1)(12);
                    in_ch_accu_nxt(194)<= in_ch_accu(194)+mac_mul( 0)(15)+mac_mul( 1)(13);
                    in_ch_accu_nxt(195)<= in_ch_accu(195)+mac_mul( 1)(14)+mac_mul( 2)(12);
                    in_ch_accu_nxt(196)<= in_ch_accu(196)+mac_mul( 1)(15)+mac_mul( 2)(13);
                    in_ch_accu_nxt(197)<= in_ch_accu(197)+mac_mul( 2)(14)+mac_mul( 3)(12);
                    in_ch_accu_nxt(198)<= in_ch_accu(198)+mac_mul( 2)(15)+mac_mul( 3)(13);
                    in_ch_accu_nxt(199)<= in_ch_accu(199)+mac_mul( 3)(14)+mac_mul( 4)(12);
                    in_ch_accu_nxt(200)<= in_ch_accu(200)+mac_mul( 3)(15)+mac_mul( 4)(13);
                    in_ch_accu_nxt(201)<= in_ch_accu(201)+mac_mul( 4)(14)+mac_mul( 5)(12);
                    in_ch_accu_nxt(202)<= in_ch_accu(202)+mac_mul( 4)(15)+mac_mul( 5)(13);
                    in_ch_accu_nxt(203)<= in_ch_accu(203)+mac_mul( 5)(14)+mac_mul( 6)(12);
                    in_ch_accu_nxt(204)<= in_ch_accu(204)+mac_mul( 5)(15)+mac_mul( 6)(13);
                    in_ch_accu_nxt(205)<= in_ch_accu(205)+mac_mul( 6)(14)+mac_mul( 7)(12);
                    in_ch_accu_nxt(206)<= in_ch_accu(206)+mac_mul( 6)(15)+mac_mul( 7)(13);
                    in_ch_accu_nxt(207)<= in_ch_accu(207)+mac_mul( 7)(14)+mac_mul( 8)(12);
                    in_ch_accu_nxt(208)<= in_ch_accu(208)+mac_mul( 7)(15)+mac_mul( 8)(13);
                    in_ch_accu_nxt(209)<= in_ch_accu(209)+mac_mul( 8)(14)+mac_mul( 9)(12);
                    in_ch_accu_nxt(210)<= in_ch_accu(210)+mac_mul( 8)(15)+mac_mul( 9)(13);
                    in_ch_accu_nxt(211)<= in_ch_accu(211)+mac_mul( 9)(14)+mac_mul(10)(12);
                    in_ch_accu_nxt(212)<= in_ch_accu(212)+mac_mul( 9)(15)+mac_mul(10)(13);
                    in_ch_accu_nxt(213)<= in_ch_accu(213)+mac_mul(10)(14)+mac_mul(11)(12);
                    in_ch_accu_nxt(214)<= in_ch_accu(214)+mac_mul(10)(15)+mac_mul(11)(13);
                    in_ch_accu_nxt(215)<= in_ch_accu(215)+mac_mul(11)(14)+mac_mul(12)(12);
                    in_ch_accu_nxt(216)<= in_ch_accu(216)+mac_mul(11)(15)+mac_mul(12)(13);
                    in_ch_accu_nxt(217)<= in_ch_accu(217)+mac_mul(12)(14)+mac_mul(13)(12);
                    in_ch_accu_nxt(218)<= in_ch_accu(218)+mac_mul(12)(15)+mac_mul(13)(13);
                    in_ch_accu_nxt(219)<= in_ch_accu(219)+mac_mul(13)(14)+mac_mul(14)(12);
                    in_ch_accu_nxt(220)<= in_ch_accu(220)+mac_mul(13)(15)+mac_mul(14)(13);
                    in_ch_accu_nxt(221)<= in_ch_accu(221)+mac_mul(14)(14)+mac_mul(15)(12);
                    in_ch_accu_nxt(222)<= in_ch_accu(222)+mac_mul(14)(15)+mac_mul(15)(13);
                    in_ch_accu_nxt(223)<= in_ch_accu(223)+mac_mul(15)(14);
                    in_ch_accu_nxt(224)<= in_ch_accu(224)+mac_mul(15)(15);
                    end if;

                end if;

                -- if in_size=32, 1 row have 2 x 16column, need to calculate both 2 column, so rowcol_cnt +1 & -1
                if (in_size(5)='1') then
                    if (rowcol_cnt(0)='1') then
                        if (in_ch_cnt=in_channel-1) then
                            rowcol_cnt_nxt <= rowcol_cnt;
                        else
                            rowcol_cnt_nxt <= rowcol_cnt - 1;
                        end if;
                    else
                        rowcol_cnt_nxt <= rowcol_cnt + 1;
                    end if;
                else
                    rowcol_cnt_nxt <= rowcol_cnt;
                end if;

                -- in_size=32, need calculate 2 time (0~15 & 16~31), so move to next input channel after 2 loop
                if (in_size(5)='1' and rowcol_cnt(0)='0') then
                    ch_cnt_nxt <= ch_cnt;
                    in_ch_cnt_nxt <= in_ch_cnt;
                else
                    ch_cnt_nxt <= ch_cnt + 1;
                    in_ch_cnt_nxt <= in_ch_cnt + 1;
                end if;
                
            when S_CONVT_addbias =>
                -- add bias to output part of input channel accumulator
                -- output part is the part that will be output at the next state
                -- it will output 2 row (0~63, 64~127)
                -- sometimes the 2 output row are different output channel, they need to add different bias
                if (rowcol_cnt=0) then
                    for i in 0 to 63 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i) + bias_buf2;
                    end loop;
                    for i in 64 to 127 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i) + bias_buf1;
                    end loop;
                else
                    for i in 0 to 127 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i) + bias_buf1;
                    end loop;
                end if;
            when S_CONVT_out => -- output state
                -- set address
                mem_addr_0 <= out_addr + out_rowcol_cnt;
                if (out_rowcol_cnt=x"FFF") then -- skip the first invalid output
                    mem_en <= '0';
                else
                    mem_en <= '1';
                end if;
                mem_RW <= write;

                out_rowcol_cnt_nxt <= out_rowcol_cnt + 1;

                if (in_size(2)='1' or in_size(3)='1' or (in_size(4)='1' and out_rowcol_cnt(0)='1') or
                    (in_size(5)='1' and out_rowcol_cnt(1 downto 0)="11")) then
                    -- shift for a row
                    for i in 0 to 191 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i+64);
                    end loop;
                    for i in 192 to 255 loop
                        in_ch_accu_nxt(i) <= (others => '0');
                    end loop;
                else
                    -- shift for 16 data (when dealing with output_size of 32 & 64)
                    for i in 0 to 47 loop
                        in_ch_accu_nxt(i) <= in_ch_accu(i+16);
                    end loop;
                end if;

                if ((in_size(2)='1' and out_rowcol_cnt(0)='0') or 
                    (in_size(3)='1' and out_rowcol_cnt(0)='0') or 
                    (in_size(4)='1' and out_rowcol_cnt(1 downto 0)="01") or
                    (in_size(5)='1' and out_rowcol_cnt(2 downto 0)="011")) then
                    -- finishing this output state

                    if ((in_size(2)='1' and rowcol_cnt(1 downto 0)=3) or
                        (in_size(3)='1' and rowcol_cnt(2 downto 0)=7) or
                        (in_size(4)='1' and rowcol_cnt(3 downto 0)=15) or
                        (in_size(5)='1' and rowcol_cnt(5 downto 0)=63)) then
                        -- finish an output channel
                        rowcol_cnt_nxt <= (others => '0');
                        in_ch_cnt_nxt <= (others => '0');
                        ch_cnt_nxt <= ch_cnt;
                        out_ch_cnt_nxt <= out_ch_cnt + 1;
                    else
                        -- haven't finish an output channel
                        -- go to calculate next row
                        rowcol_cnt_nxt <= rowcol_cnt + 1;
                        in_ch_cnt_nxt <= (others => '0');
                        ch_cnt_nxt <= ch_cnt - in_channel;
                        out_ch_cnt_nxt <= out_ch_cnt;
                    end if;
                end if;
            when S_CONVT_endbias =>
                -- need additional state to output the last row
                for i in 0 to 63 loop
                    in_ch_accu_nxt(i) <= in_ch_accu(i) + bias_buf1;
                end loop;
            when S_CONVT_endout =>
                -- need additional state to output the last row

                -- set address
                mem_addr_0 <= out_addr + out_rowcol_cnt;
                mem_en <= '1';
                mem_RW <= write;

                out_rowcol_cnt_nxt <= out_rowcol_cnt + 1;

                for i in 0 to 47 loop
                    in_ch_accu_nxt(i) <= in_ch_accu(i+16);
                end loop;
            when S_END =>

            when others =>
                null;
        end case;
    end process;

    ch_cnt_mod16 <= to_integer(ch_cnt(3 downto 0));
    out_ch_cnt_mod16 <= to_integer(out_ch_cnt(3 downto 0));
    -- 16-1 multiplexer to select data from memory
    mem_in_sel16_out_ch <= resize(signed(mem_in(out_ch_cnt_mod16*16+15 downto out_ch_cnt_mod16*16)&"00000"),32);

    -- multiplication result for MULADD layer
    muladd_mul0 : for i in 0 to 15 generate
        muladd_mul(i) <= input_buf(i)*shf_reg1(0);
    end generate muladd_mul0;

    process(clock)
    begin
        if (rising_edge(clock)) then
            if (reset='0') then
                input_buf <= (others => (others => '0'));
                wgt_buf <= (others => (others => '0'));
                bias_buf1 <= (others => '0');
                bias_buf2 <= (others => '0');
                in_ch_cnt <= (others => '0');
                out_ch_cnt <= (others => '0');
                ch_cnt <= (others => '0');
                rowcol_cnt <= (others => '0');
                in_ch_accu <= (others => (others => '0'));
                mem_read_cnt <= (others => '0');
                shf_reg1 <= (others => (others => '0'));
                shf_reg2 <= (others => (others => '0'));
                out_rowcol_cnt <= (others => '0');
            else
                input_buf <= input_buf_nxt;
                wgt_buf <= wgt_buf_nxt;
                bias_buf1 <= bias_buf1_nxt;
                bias_buf2 <= bias_buf2_nxt;
                in_ch_cnt <= in_ch_cnt_nxt;
                out_ch_cnt <= out_ch_cnt_nxt;
                ch_cnt <= ch_cnt_nxt;
                rowcol_cnt <= rowcol_cnt_nxt;
                in_ch_accu <= in_ch_accu_nxt;
                mem_read_cnt <= mem_read_cnt_nxt;
                shf_reg1 <= shf_reg1_nxt;
                shf_reg2 <= shf_reg2_nxt;
                out_rowcol_cnt <= out_rowcol_cnt_nxt;
            end if;
        end if;
    end process;
    ----------------------------------------------------------------------------

    -- MAC ---------------------------------------------------------------------
    process(input_buf, state)
    begin
        case (state) is
            when S_LINEAR_EX =>
                for i in 0 to 15 loop
                    mac_in(i) <= input_buf;
                end loop;
            when S_CONV3_ex =>
                mac_in(0) <= (0 => (others => '0'), 1 => (others => '0'), 2 => (others => '0'),
                              3 => (others => '0'), 4 => input_buf(0)   , 5 => input_buf(1)   ,
                              6 => (others => '0'), 7 => input_buf(4)   , 8 => input_buf(5)   ,
                              others => (others => '0'));
                mac_in(1) <= (0 => (others => '0'), 1 => (others => '0'), 2 => (others => '0'),
                              3 => input_buf(0)   , 4 => input_buf(1)   , 5 => input_buf(2)   ,
                              6 => input_buf(4)   , 7 => input_buf(5)   , 8 => input_buf(6)   ,
                              others => (others => '0'));
                mac_in(2) <= (0 => (others => '0'), 1 => (others => '0'), 2 => (others => '0'),
                              3 => input_buf(1)   , 4 => input_buf(2)   , 5 => input_buf(3)   ,
                              6 => input_buf(5)   , 7 => input_buf(6)   , 8 => input_buf(7)   ,
                              others => (others => '0'));
                mac_in(3) <= (0 => (others => '0'), 1 => (others => '0'), 2 => (others => '0'),
                              3 => input_buf(2)   , 4 => input_buf(3)   , 5 => (others => '0'),
                              6 => input_buf(6)   , 7 => input_buf(7)   , 8 => (others => '0'),
                              others => (others => '0'));
                mac_in(4) <= (0 => (others => '0'), 1 => input_buf(0)   , 2 => input_buf(1)   ,
                              3 => (others => '0'), 4 => input_buf(4)   , 5 => input_buf(5)   ,
                              6 => (others => '0'), 7 => input_buf(8)   , 8 => input_buf(9)   ,
                              others => (others => '0'));
                mac_in(5) <= (0 => input_buf(0)   , 1 => input_buf(1)   , 2 => input_buf(2)   ,
                              3 => input_buf(4)   , 4 => input_buf(5)   , 5 => input_buf(6)   ,
                              6 => input_buf(8)   , 7 => input_buf(9)   , 8 => input_buf(10)  ,
                              others => (others => '0'));
                mac_in(6) <= (0 => input_buf(1)   , 1 => input_buf(2)   , 2 => input_buf(3)   ,
                              3 => input_buf(5)   , 4 => input_buf(6)   , 5 => input_buf(7)   ,
                              6 => input_buf(9)   , 7 => input_buf(10)  , 8 => input_buf(11)  ,
                              others => (others => '0'));
                mac_in(7) <= (0 => input_buf(2)   , 1 => input_buf(3)   , 2 => (others => '0'),
                              3 => input_buf(6)   , 4 => input_buf(7)   , 5 => (others => '0'),
                              6 => input_buf(10)  , 7 => input_buf(11)  , 8 => (others => '0'),
                              others => (others => '0'));
                mac_in(8) <= (0 => (others => '0'), 1 => input_buf(4)   , 2 => input_buf(5)   ,
                              3 => (others => '0'), 4 => input_buf(8)   , 5 => input_buf(9)   ,
                              6 => (others => '0'), 7 => input_buf(12)  , 8 => input_buf(13)  ,
                              others => (others => '0'));
                mac_in(9) <= (0 => input_buf(4)   , 1 => input_buf(5)   , 2 => input_buf(6)   ,
                              3 => input_buf(8)   , 4 => input_buf(9)   , 5 => input_buf(10)  ,
                              6 => input_buf(12)  , 7 => input_buf(13)  , 8 => input_buf(14)  ,
                              others => (others => '0'));
                mac_in(10)<= (0 => input_buf(5)   , 1 => input_buf(6)   , 2 => input_buf(7)   ,
                              3 => input_buf(9)   , 4 => input_buf(10)  , 5 => input_buf(11)  ,
                              6 => input_buf(13)  , 7 => input_buf(14)  , 8 => input_buf(15)  ,
                              others => (others => '0'));
                mac_in(11)<= (0 => input_buf(6)   , 1 => input_buf(7)   , 2 => (others => '0'),
                              3 => input_buf(10)  , 4 => input_buf(11)  , 5 => (others => '0'),
                              6 => input_buf(14)  , 7 => input_buf(15)  , 8 => (others => '0'),
                              others => (others => '0'));
                mac_in(12)<= (0 => (others => '0'), 1 => input_buf(8)   , 2 => input_buf(9)   ,
                              3 => (others => '0'), 4 => input_buf(12)  , 5 => input_buf(13)   ,
                              6 => (others => '0'), 7 => (others => '0'), 8 => (others => '0'),
                              others => (others => '0'));
                mac_in(13)<= (0 => input_buf(8)   , 1 => input_buf(9)   , 2 => input_buf(10)  ,
                              3 => input_buf(12)  , 4 => input_buf(13)  , 5 => input_buf(14)  ,
                              6 => (others => '0'), 7 => (others => '0'), 8 => (others => '0'),
                              others => (others => '0'));
                mac_in(14)<= (0 => input_buf(9)   , 1 => input_buf(10)  , 2 => input_buf(11)  ,
                              3 => input_buf(13)  , 4 => input_buf(14)  , 5 => input_buf(15)  ,
                              6 => (others => '0'), 7 => (others => '0'), 8 => (others => '0'),
                              others => (others => '0'));
                mac_in(15)<= (0 => input_buf(10)  , 1 => input_buf(11)  , 2 => (others => '0'),
                              3 => input_buf(14)  , 4 => input_buf(15)  , 5 => (others => '0'),
                              6 => (others => '0'), 7 => (others => '0'), 8 => (others => '0'),
                              others => (others => '0'));
            when S_CONVT_ex =>
                for i in 0 to 15 loop
                    mac_in(i) <= (others => input_buf(i));
                end loop;
            when others =>
                for i in 0 to 15 loop
                    mac_in(i) <= (others => (others => '0'));
                end loop;
        end case;
    end process;

    macwgt0: for i in 0 to 15 generate
        mac_wgt(i) <= wgt_buf;
    end generate;

    MAC_gen : for i in 0 to 15 generate
        MAC0: mac4x4 port map (
            in0  => mac_in(i)( 0), in1  => mac_in(i)( 1), in2  => mac_in(i)( 2), in3  => mac_in(i)( 3),
            in4  => mac_in(i)( 4), in5  => mac_in(i)( 5), in6  => mac_in(i)( 6), in7  => mac_in(i)( 7),
            in8  => mac_in(i)( 8), in9  => mac_in(i)( 9), in10 => mac_in(i)(10), in11 => mac_in(i)(11),
            in12 => mac_in(i)(12), in13 => mac_in(i)(13), in14 => mac_in(i)(14), in15 => mac_in(i)(15),
            wgt0  => mac_wgt(i)( 0), wgt1  => mac_wgt(i)( 1), wgt2  => mac_wgt(i)( 2), wgt3  => mac_wgt(i)( 3),
            wgt4  => mac_wgt(i)( 4), wgt5  => mac_wgt(i)( 5), wgt6  => mac_wgt(i)( 6), wgt7  => mac_wgt(i)( 7),
            wgt8  => mac_wgt(i)( 8), wgt9  => mac_wgt(i)( 9), wgt10 => mac_wgt(i)(10), wgt11 => mac_wgt(i)(11),
            wgt12 => mac_wgt(i)(12), wgt13 => mac_wgt(i)(13), wgt14 => mac_wgt(i)(14), wgt15 => mac_wgt(i)(15),
            mul0  => mac_mul(i)( 0), mul1  => mac_mul(i)( 1), mul2  => mac_mul(i)( 2), mul3  => mac_mul(i)( 3),
            mul4  => mac_mul(i)( 4), mul5  => mac_mul(i)( 5), mul6  => mac_mul(i)( 6), mul7  => mac_mul(i)( 7),
            mul8  => mac_mul(i)( 8), mul9  => mac_mul(i)( 9), mul10 => mac_mul(i)(10), mul11 => mac_mul(i)(11),
            mul12 => mac_mul(i)(12), mul13 => mac_mul(i)(13), mul14 => mac_mul(i)(14), mul15 => mac_mul(i)(15),
            MAC => mac_out(i)
        );
    end generate;
    ----------------------------------------------------------------------------

    
end  RTL;