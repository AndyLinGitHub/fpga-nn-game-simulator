--------------------------------------------------------------------------------
-- This module generate output RGB data to hdmi 
-- 
-- input: 
-- 		clock: clock
-- 		rst_n: active low reset 
-- 		h_total_cnt: horizontal counter from hdmi_timing_gen module 
-- 		v_total_cnt: vertical   counter from hdmi_timing_gen module 
-- 		bram_dout: data fetch from memory 
-- output: 
-- 		pixel_R: RGB output (0~255)
-- 		pixel_G: RGB output (0~255)
-- 		pixel_B: RGB output (0~255)
--		bram_addr: address of memory 
-- 
-- We use 640x480 60Hz
-- 
-- The frame is valid when 160<h_counter<640 & 45<v_counter<525
-- I choose to output our image when 256<h_counter<511 & 256<v_counter<511, otherwise output black
-- Our image is 64x64, so we scale it up it to 256x256
--
-- When h_counter=16, it will start fetching data of this row from memory 
-- 		each row have 64 data, memory data width is 16, so it take 4 time to fetch a row 
-- 		fetch RGB take 12 time, each fetch take 3 clocks, so total 36 clocks fetching a row 
-- 
-- h_counter=16~51: fetch data of a row 
-- h_counter=256~511 (and v_counter=256~511): output data
-- 
-- Yu-An Lin & Tzu-Chi Huang
--------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity hdmi_data_gen is
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
end hdmi_data_gen;

architecture RTL of hdmi_data_gen is
	-- output when when 256<h_counter<511 & 256<v_counter<511 
	constant v_start : unsigned(11 downto 0) := to_unsigned(256,12);
	constant h_start : unsigned(11 downto 0) := to_unsigned(256,12);
	constant v_end : unsigned(11 downto 0) := to_unsigned(511,12);
	constant h_end : unsigned(11 downto 0) := to_unsigned(511,12);

	-- starting address of RGB data
	constant R_addr : unsigned(15 downto 0) := to_unsigned(51456,16);
	constant G_addr : unsigned(15 downto 0) := to_unsigned(51712,16);
	constant B_addr : unsigned(15 downto 0) := to_unsigned(51968,16);

	-- RGB output of color black
	constant Black : std_logic_vector(7 downto 0) := x"00";

	type array_64x8 is array (63 downto 0) of std_logic_vector(7 downto 0);
	signal R_reg, R_reg_nxt : array_64x8; --64x8
	signal G_reg, G_reg_nxt : array_64x8; --64x8
	signal B_reg, B_reg_nxt : array_64x8; --64x8

	signal read_cnt : unsigned(1 downto 0);

	-- FSM
    type state_type is (IDLE, 
    					R_set_addr, R_wait_addr, R_read_mem,
    					G_set_addr, G_wait_addr, G_read_mem,
    					B_set_addr, B_wait_addr, B_read_mem); -- FSM
    signal state : state_type; -- FSM
begin

	-- output frame between v_cnt=256~511 and h_cnt=256~511, otherwise output black
	pixel_R <= R_reg(0) when(v_total_cnt(11 downto 8)=1 and h_total_cnt(11 downto 8)=1)
				else Black;
	pixel_G <= G_reg(0) when(v_total_cnt(11 downto 8)=1 and h_total_cnt(11 downto 8)=1)
				else Black;
	pixel_B <= B_reg(0) when(v_total_cnt(11 downto 8)=1 and h_total_cnt(11 downto 8)=1)
				else Black;

	process (clk)
	begin
		if (rising_edge(clk)) then
			if (rst_n='0') then
				state <= IDLE;
				read_cnt <= (others => '0');
			else
				state <= state;
				read_cnt <= read_cnt;
				case (state) is
					when IDLE => -- IDLE
						if (h_total_cnt=16) then
							state <= R_set_addr;
						end if;
						read_cnt <= (others => '0');
					when R_set_addr => -- set address
						state <= R_wait_addr;
					when R_wait_addr => -- wait for memory
						state <= R_read_mem;
					when R_read_mem => -- store data to register
						-- fetch 4 time for a row
						if (read_cnt=3) then
							state <= G_set_addr;
						else
							state <= R_set_addr;
						end if;
						read_cnt <= read_cnt+1;
					when G_set_addr => -- set address
						state <= G_wait_addr;
					when G_wait_addr => -- wait for memory
						state <= G_read_mem;
					when G_read_mem => -- store data to register
						-- fetch 4 time for a row
						if (read_cnt=3) then
							state <= B_set_addr;
						else
							state <= G_set_addr;
						end if;
						read_cnt <= read_cnt+1;
					when B_set_addr => -- set address
						state <= B_wait_addr;
					when B_wait_addr => -- wait for memory
						state <= B_read_mem;
					when B_read_mem =>-- store data to register
						-- fetch 4 time for a row
						if (read_cnt=3) then
							state <= IDLE;
						else
							state <= B_set_addr;
						end if;
						read_cnt <= read_cnt+1;
					when others =>
						state <= IDLE;
				end case;
			end if;
		end if;
	end process;

	process (state, h_total_cnt, v_total_cnt, R_reg, G_reg, B_reg,
			 read_cnt, bram_dout)
	begin
		R_reg_nxt <= R_reg;
		G_reg_nxt <= G_reg;
		B_reg_nxt <= B_reg;
		bram_addr <= (others => '0');
		case (state) is
			when IDLE =>
				-- output when 256<h_counter<511 & 256<v_counter<511
				-- shift the register to output
				if (v_total_cnt(11 downto 8)=1 and h_total_cnt(11 downto 8)=1 and h_total_cnt(1 downto 0)=3) then
					R_reg_nxt(63) <= x"00";
					G_reg_nxt(63) <= x"00";
					B_reg_nxt(63) <= x"00";
					for i in 62 downto 0 loop
						R_reg_nxt(i) <= R_reg(i+1);
						G_reg_nxt(i) <= G_reg(i+1);
						B_reg_nxt(i) <= B_reg(i+1);
					end loop;
				end if;
			when R_set_addr =>
				-- set address
				bram_addr <= std_logic_vector(R_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
			when R_wait_addr =>
				-- wait
				bram_addr <= std_logic_vector(R_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
			when R_read_mem =>
				bram_addr <= std_logic_vector(R_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
				-- shift the register to load data
				for i in 15 downto 0 loop
					R_reg_nxt(i+48) <= bram_dout(i*16+14 downto i*16+7);
				end loop;
				for i in 47 downto 0 loop
					R_reg_nxt(i) <= R_reg(i+16);
				end loop;
			when G_set_addr =>
				-- set address
				bram_addr <= std_logic_vector(G_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
			when G_wait_addr =>
				-- wait
				bram_addr <= std_logic_vector(G_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
			when G_read_mem =>
				bram_addr <= std_logic_vector(G_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
				-- shift the register to load data
				for i in 15 downto 0 loop
					G_reg_nxt(i+48) <= bram_dout(i*16+14 downto i*16+7);
				end loop;
				for i in 47 downto 0 loop
					G_reg_nxt(i) <= G_reg(i+16);
				end loop;
			when B_set_addr =>
				-- set address
				bram_addr <= std_logic_vector(B_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
			when B_wait_addr =>
				-- wait
				bram_addr <= std_logic_vector(B_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
			when B_read_mem =>
				bram_addr <= std_logic_vector(B_addr + unsigned(v_total_cnt(7 downto 2)&read_cnt));
				-- shift the register to load data
				for i in 15 downto 0 loop
					B_reg_nxt(i+48) <= bram_dout(i*16+14 downto i*16+7);
				end loop;
				for i in 47 downto 0 loop
					B_reg_nxt(i) <= B_reg(i+16);
				end loop;
			when others =>

		end case;
	end process;
	
	process (clk)
	begin
		if (rising_edge(clk)) then
			if (rst_n='0') then
				R_reg <= (others => (others => '0'));
				G_reg <= (others => (others => '0'));
				B_reg <= (others => (others => '0'));
			else
				R_reg <= R_reg_nxt;
				G_reg <= G_reg_nxt;
				B_reg <= B_reg_nxt;
			end if;
		end if;
	end process;

end RTL;