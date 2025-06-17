--------------------------------------------------------------------------------
-- reference: https://blog.csdn.net/zhoutaopower/article/details/113485579
-- 
-- This module generate timing signal for outputing hdmi
-- 
-- translate from verilog to vhdl: chatgpt & Tzu-Chi Huang
--------------------------------------------------------------------------------

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity hdmi_timing_gen is
    Port (
        clk         : in  STD_LOGIC;
        rst_n       : in  STD_LOGIC;
        vsync       : out STD_LOGIC;
        hsync       : out STD_LOGIC;
        video_valid : out STD_LOGIC;
        h_total_cnt0 : out unsigned(11 downto 0);
        v_total_cnt0 : out unsigned(11 downto 0)
    );
end hdmi_timing_gen;

architecture RTL of hdmi_timing_gen is


    -- Timing parameters (conditional)
    constant H_ACTIVE : integer := 640;

    constant H_FP     : integer := 16;

    constant H_SYNC   : integer := 96;

    constant H_BP     : integer := 48;

    constant V_ACTIVE : integer := 480;

    constant V_FP     : integer := 10;

    constant V_SYNC   : integer := 2;

    constant V_BP     : integer := 33;

    constant HS_POL   : std_logic := '1';
    constant VS_POL   : std_logic := '1';

    constant H_TOTAL  : integer := H_ACTIVE + H_FP + H_SYNC + H_BP;
    constant V_TOTAL  : integer := V_ACTIVE + V_FP + V_SYNC + V_BP;

    signal h_total_cnt : unsigned(11 downto 0);
    signal v_total_cnt : unsigned(11 downto 0);

    signal hs_reg      : std_logic;
    signal hs_reg_d0   : std_logic;
    signal vs_reg      : std_logic;
    signal vs_reg_d0   : std_logic;

    signal h_valid     : std_logic;
    signal v_valid     : std_logic;
    signal video_valid_reg     : std_logic;
    signal video_valid_reg_d0  : std_logic;

begin
    h_total_cnt0 <= h_total_cnt;
    v_total_cnt0 <= v_total_cnt;
    
    
    -- Horizontal Counter
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            h_total_cnt <= (others => '0');
	        else
	            if h_total_cnt = to_unsigned(H_TOTAL - 1, 12) then
	                h_total_cnt <= (others => '0');
	            else
	                h_total_cnt <= h_total_cnt + 1;
	            end if;
	        end if;
        end if;
    end process;

    -- Vertical Counter
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            v_total_cnt <= (others => '0');
	        else
	            if h_total_cnt = to_unsigned(H_FP - 1, 12) then
	                if v_total_cnt = to_unsigned(V_TOTAL - 1, 12) then
	                    v_total_cnt <= (others => '0');
	                else
	                    v_total_cnt <= v_total_cnt + 1;
	                end if;
	            else
	            	v_total_cnt <= v_total_cnt;
	            end if;
	        end if;
        end if;
    end process;

    -- HSYNC Signal
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            hs_reg <= not HS_POL;
	        else
	            if h_total_cnt = to_unsigned(H_FP - 1, 12) then
	                hs_reg <= HS_POL;
	            elsif h_total_cnt = to_unsigned(H_FP + H_SYNC - 1, 12) then
	                hs_reg <= not HS_POL;
	            else
	            	hs_reg <= hs_reg;
	            end if;
	        end if;
        end if;
    end process;

    -- VSYNC Signal
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            vs_reg <= not VS_POL;
	        else
	            if (v_total_cnt = to_unsigned(V_FP - 1, 12)) and (h_total_cnt = to_unsigned(H_FP - 1, 12)) then
	                vs_reg <= VS_POL;
	            elsif (v_total_cnt = to_unsigned(V_FP + V_SYNC - 1, 12)) and (h_total_cnt = to_unsigned(H_FP - 1, 12)) then
	                vs_reg <= not VS_POL;
	            else
	            	vs_reg <= vs_reg;
	            end if;
	        end if;
        end if;
    end process;

    -- Horizontal Valid Signal
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            h_valid <= '0';
	        else
	            if h_total_cnt = to_unsigned(H_FP + H_SYNC + H_BP - 1, 12) then
	                h_valid <= '1';
	            elsif h_total_cnt = to_unsigned(H_TOTAL - 1, 12) then
	                h_valid <= '0';
	            else
	            	h_valid <= h_valid;
	            end if;
	        end if;
        end if;
    end process;

    -- Vertical Valid Signal
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            v_valid <= '0';
	        else
	            if (v_total_cnt = to_unsigned(V_FP + V_SYNC + V_BP - 1, 12)) and (h_total_cnt = to_unsigned(H_FP - 1, 12)) then
	                v_valid <= '1';
	            elsif (v_total_cnt = to_unsigned(V_TOTAL - 1, 12)) and (h_total_cnt = to_unsigned(H_FP - 1, 12)) then
	                v_valid <= '0';
	            else
	            	v_valid <= v_valid;
	            end if;
	        end if;
        end if;
    end process;

    -- Delay one pixel clock for hsync
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            hs_reg_d0 <= '0';
	        else
	            hs_reg_d0 <= hs_reg;
	        end if;
        end if;
    end process;
    hsync <= hs_reg_d0;

    -- Delay one pixel clock for vsync
    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            vs_reg_d0 <= '0';
	        else
	            vs_reg_d0 <= vs_reg;
	        end if;
        end if;
    end process;
    vsync <= vs_reg_d0;

    -- Video Valid
    video_valid_reg <= v_valid and h_valid;

    process(clk)
    begin
    	if (rising_edge(clk)) then
	        if rst_n = '0' then
	            video_valid_reg_d0 <= '0';
	        else
	            video_valid_reg_d0 <= video_valid_reg;
	        end if;
        end if;
    end process;
    video_valid <= video_valid_reg_d0;

end RTL;
