----------------------------------------------------------------------------------
-- This is the top module of the whole design 
-- 
-- input: 
-- 		clock: 10ns clock 
-- 		reset: active low reset 
-- 		button_u: up button 
-- 		button_d: down button 
-- 		button_l: left button 
-- 		button_r: right button 
-- 		button_c: central button 
-- output: 
-- 		TMDS_clk_p : hdmi signal
--	  	TMDS_clk_n : hdmi signal
--	  	TMDS_data_p: hdmi signal
--	  	TMDS_data_n: hdmi signal 
-- 
-- submodule: 
-- 		NNA: neural network accelerator: calculation of neural network 
-- 		bram: dual port block RAM on fpga (generated by vivado) 
-- 		hdmi_timing_gen: generate hdmi timing signal 
-- 		hdmi_data_gen: generate RGB data for output 
-- 		hdmi_rgb2dvi: generate hdmi output signal, (generated by vivado) 
----------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
    Port (clock   : in  std_logic;
	      reset   : in  std_logic;
	      button_u: in  std_logic;
	      button_d: in  std_logic;
	      button_l: in  std_logic;
	      button_r: in  std_logic;
	      button_c: in  std_logic;
	      TMDS_clk_p   : out std_logic;
		  TMDS_clk_n   : out std_logic;
		  TMDS_data_p  : out std_logic_vector(2 downto 0);
		  TMDS_data_n  : out std_logic_vector(2 downto 0)
	      );
end top;

architecture RTL of top is
	
	component NNA  is
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
	end component;

	component bram is
	port (
	    BRAM_PORTA_addr : in STD_LOGIC_VECTOR ( 15 downto 0 );
	    BRAM_PORTA_clk : in STD_LOGIC;
	    BRAM_PORTA_din : in STD_LOGIC_VECTOR ( 255 downto 0 );
	    BRAM_PORTA_dout : out STD_LOGIC_VECTOR ( 255 downto 0 );
	    BRAM_PORTA_en : in STD_LOGIC;
	    BRAM_PORTA_we : in STD_LOGIC_VECTOR ( 0 to 0 );

	    BRAM_PORTB_addr : in STD_LOGIC_VECTOR ( 15 downto 0 );
	    BRAM_PORTB_clk : in STD_LOGIC;
	    BRAM_PORTB_din : in STD_LOGIC_VECTOR ( 255 downto 0 );
	    BRAM_PORTB_dout : out STD_LOGIC_VECTOR ( 255 downto 0 );
	    BRAM_PORTB_en : in STD_LOGIC;
	    BRAM_PORTB_we : in STD_LOGIC_VECTOR ( 0 to 0 )
	);
	end component;

	component hdmi_timing_gen is
    Port (
        clk         : in  STD_LOGIC;
        rst_n       : in  STD_LOGIC;
        vsync       : out STD_LOGIC;
        hsync       : out STD_LOGIC;
        video_valid : out STD_LOGIC;
        h_total_cnt0 : out unsigned(11 downto 0);
        v_total_cnt0 : out unsigned(11 downto 0)
    );
	end component;

	component hdmi_data_gen is
    Port (
        clk      : in  std_logic;
        rst_n    : in  std_logic;
        h_total_cnt : in unsigned(11 downto 0);
        v_total_cnt : in unsigned(11 downto 0);
        bram_dout : in std_logic_vector(255 downto 0);

        pixel_R  : out std_logic_vector(7 downto 0);
        pixel_G  : out std_logic_vector(7 downto 0);
        pixel_B  : out std_logic_vector(7 downto 0);
        bram_addr : out std_logic_vector(15 downto 0)
    );
	end component;

	component hdmi_rgb2dvi is
    port (
      -------------------------------------------------------------------------
      -- video stream in
      -------------------------------------------------------------------------
      RGB_active_video : in std_logic;
      RGB_data         : in std_logic_vector(23 downto 0);
      RGB_hsync        : in std_logic;
      RGB_vsync        : in std_logic;

      -------------------------------------------------------------------------
      -- TMDS out
      -------------------------------------------------------------------------
      TMDS_clk_p       : out std_logic;
      TMDS_clk_n       : out std_logic;
      TMDS_data_p      : out std_logic_vector(2 downto 0);
      TMDS_data_n      : out std_logic_vector(2 downto 0);

      -------------------------------------------------------------------------
      -- clocks / control
      -------------------------------------------------------------------------
      clk_in1          : in std_logic;    -- same pixel clock used everywhere
      clk_out1         : out std_logic;   -- recovered/phase-shifted clock
      locked           : out std_logic;    -- MMCM/DPLL lock indicator
      reset            : in std_logic
    );
    end component;

    -- NNA signal
	signal NNA_start   : std_logic;
    signal NNA_button  : std_logic_vector( 8*16-1 downto 0);
    signal NNA_mem_in  : std_logic_vector(16*16-1 downto 0);
    signal NNA_mem_en  : std_logic;
    signal NNA_mem_RW  : std_logic;
    signal NNA_mem_out : std_logic_vector(16*16-1 downto 0);
    signal NNA_mem_addr: std_logic_vector(15 downto 0);
    signal NNA_finish  : std_logic;

    -- button register
    signal button_u_FF0 : std_logic;
    signal button_u_FF1 : std_logic;
    signal button_d_FF0 : std_logic;
    signal button_d_FF1 : std_logic;
    signal button_l_FF0 : std_logic;
    signal button_l_FF1 : std_logic;
    signal button_r_FF0 : std_logic;
    signal button_r_FF1 : std_logic;

    -- BRAM signal
    signal bram_en_a, bram_en_b : std_logic;
    signal bram_we_a, bram_we_b : std_logic_vector(0 downto 0);
	signal bram_addr_a, bram_addr_b : std_logic_vector(15 downto 0);
	signal bram_din_a, bram_dout_a, bram_din_b, bram_dout_b : std_logic_vector(255 downto 0);

	-- HDMI signal
	signal pixel_R, pixel_G, pixel_B : std_logic_vector(7 downto 0);
	signal vsync, hsync, video_valid : std_logic;
	signal rgb_data                  : std_logic_vector(23 downto 0);
	signal reset_hdmi_rgb2dvi        : std_logic;
	signal locked               : std_logic;
	signal clk_out1             : std_logic;
	signal h_total_cnt, v_total_cnt : unsigned(11 downto 0);

	-- start running signal
	signal NNA_start_reg : std_logic;
begin
	
	-- BRAM --------------------------------------------------------------------
	-- port A connected to NNA 
	-- port B connected to hdmi_data_gen
	bram0 : bram port map(
	    BRAM_PORTA_clk => clk_out1,
	    BRAM_PORTA_en => bram_en_a,
	    BRAM_PORTA_we => bram_we_a,
	    BRAM_PORTA_addr => bram_addr_a,
	    BRAM_PORTA_din => bram_din_a,
	    BRAM_PORTA_dout => bram_dout_a,

	    BRAM_PORTB_clk => clk_out1,
	    BRAM_PORTB_en => bram_en_b,
	    BRAM_PORTB_we => bram_we_b,
	    BRAM_PORTB_addr => bram_addr_b,
	    BRAM_PORTB_din => bram_din_b,
	    BRAM_PORTB_dout => bram_dout_b
    );

	-- port b always read
	bram_en_b <= '1';
	bram_we_b <= "0";
	bram_din_b <= (others => '0');


    hdmi_time : hdmi_timing_gen port map(
        clk         => clk_out1,
        rst_n       => locked,
        vsync       => vsync,
        hsync       => hsync,
        video_valid => video_valid,
        h_total_cnt0 => h_total_cnt,
        v_total_cnt0 => v_total_cnt
    );


    hdmi_data : hdmi_data_gen port map(
        clk      	=> clk_out1,
        rst_n    	=> locked,
        h_total_cnt => h_total_cnt,
        v_total_cnt => v_total_cnt,
        bram_dout   => bram_dout_b,

        pixel_R  => pixel_R,
        pixel_G  => pixel_G,
        pixel_B  => pixel_B,
        bram_addr => bram_addr_b
    );


    ---------------------------------------------------------------------------
	-- simple glue logic
	---------------------------------------------------------------------------
	rgb_data       <= pixel_R & pixel_B & pixel_G;  -- {R[23:16], B[15:8], G[7:0]}
	reset_hdmi_rgb2dvi  <= not reset;                    -- hdmi_rgb2dvi expects active-high


    hdmi : hdmi_rgb2dvi port map (
	  	-------------------------------------------------------------------------
	  	-- video stream in
		-------------------------------------------------------------------------
		RGB_active_video => video_valid,
		RGB_data         => rgb_data,
		RGB_hsync        => hsync,
		RGB_vsync        => vsync,

		-------------------------------------------------------------------------
		-- TMDS out
		-------------------------------------------------------------------------
		TMDS_clk_p       => TMDS_clk_p,
		TMDS_clk_n       => TMDS_clk_n,
		TMDS_data_p      => TMDS_data_p,
		TMDS_data_n      => TMDS_data_n,

		-------------------------------------------------------------------------
		-- clocks / control
		-------------------------------------------------------------------------
		clk_in1          => clock,        -- same pixel clock used everywhere
		clk_out1         => clk_out1,       -- recovered/phase-shifted clock
		locked           => locked,         -- MMCM/DPLL lock indicator
		reset            => reset_hdmi_rgb2dvi
	);

    -- connect NNA signal to BRAM port A
    bram_en_a <= NNA_mem_en;
    bram_we_a <= (0 => NNA_mem_RW);
    bram_addr_a <= NNA_mem_addr;
    bram_din_a <= NNA_mem_out;
    NNA_mem_in <= bram_dout_a;

	NNA0 : NNA port map (
		clock => clk_out1,
		reset => locked,
		start => NNA_start,
		button => NNA_button,
        mem_in => NNA_mem_in,
        mem_en => NNA_mem_en,
        mem_RW => NNA_mem_RW,
        mem_out => NNA_mem_out,
        mem_addr => NNA_mem_addr,
        finish => NNA_finish
	);

	-- button ------------------------------------------------------------------
	-- convert button to Q2.13 format (only 1 or 0)
	NNA_button( 15 downto   0) <= "00"&button_d_FF0&"0000000000000";
	NNA_button( 31 downto  16) <= "00"&button_d_FF1&"0000000000000";
	NNA_button( 47 downto  32) <= "00"&button_l_FF0&"0000000000000";
	NNA_button( 63 downto  48) <= "00"&button_l_FF1&"0000000000000";
	NNA_button( 79 downto  64) <= "00"&button_u_FF0&"0000000000000";
	NNA_button( 95 downto  80) <= "00"&button_u_FF1&"0000000000000";
	NNA_button(111 downto  96) <= "00"&button_r_FF0&"0000000000000";
	NNA_button(127 downto 112) <= "00"&button_r_FF1&"0000000000000";


	process (clk_out1)
	begin
		if (rising_edge(clk_out1)) then
			if (reset='0') then
				button_u_FF0 <= '0';
				button_u_FF1 <= '0';
				button_d_FF0 <= '0';
				button_d_FF1 <= '0';
				button_l_FF0 <= '0';
				button_l_FF1 <= '0';
				button_r_FF0 <= '0';
				button_r_FF1 <= '0';
			else
				button_u_FF0 <= button_u;
				button_u_FF1 <= button_u_FF0;
				button_d_FF0 <= button_d;
				button_d_FF1 <= button_d_FF0;
				button_l_FF0 <= button_l;
				button_l_FF1 <= button_l_FF0;
				button_r_FF0 <= button_r;
				button_r_FF1 <= button_r_FF0;
			end if;
		end if;
	end process;


	-- start signal ------------------------------------------------------------
	-- when press central button, start running the system
	NNA_start <= NNA_start_reg;
	process (clk_out1)
	begin
		if (rising_edge(clk_out1)) then
			if (reset='0') then
				NNA_start_reg <= '0';
			else
				if (button_c='1') then
					NNA_start_reg <= '1';
				else
					NNA_start_reg <= NNA_start_reg;
				end if;
			end if;
		end if;
	end process;

end RTL;

