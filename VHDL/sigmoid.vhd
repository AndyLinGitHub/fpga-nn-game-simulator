----------------------------------------------------------------------------------
-- This design calculate sigmoid function

-- input is a 32 bit signed Q13.18 
-- output will be 16 bit signed(but always>0) Q0.15 (0 ~ 1)

-- It use 16 linear segment to approximate sigmoid function

-- design by Tzu-Chi Huang
----------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sigmoid is
    Port (in0  : in  signed(31 downto 0); --Q13.18
    	  out0 : out signed(15 downto 0)); --Q0.15
end sigmoid;

architecture RTL of sigmoid is

	signal x0_00772177 : signed(31 downto 0); -- x*0.00772177
	signal x0_01526535 : signed(31 downto 0); -- x*0.01526535
	signal x0_02983683 : signed(31 downto 0); -- x*0.02983683
	signal x0_05704100 : signed(31 downto 0); -- x*0.05704100
	signal x0_10457516 : signed(31 downto 0); -- x*0.10457516
	signal x0_17777778 : signed(31 downto 0); -- x*0.17777778
	signal x0_26666667 : signed(31 downto 0); -- x*0.26666667
	signal x0_33333333 : signed(31 downto 0); -- x*0.33333333

	signal s0_00772177 : std_logic_vector(15 downto 0); -- 0.00772177
	signal s0_01526535 : std_logic_vector(15 downto 0); -- 0.01526535
	signal s0_02983683 : std_logic_vector(15 downto 0); -- 0.02983683
	signal s0_05704100 : std_logic_vector(15 downto 0); -- 0.05704100
	signal s0_10457516 : std_logic_vector(15 downto 0); -- 0.10457516
	signal s0_17777778 : std_logic_vector(15 downto 0); -- 0.17777778
	signal s0_26666667 : std_logic_vector(15 downto 0); -- 0.26666667
	signal s0_33333333 : std_logic_vector(15 downto 0); -- 0.33333333

	signal s0_96522185 : std_logic_vector(15 downto 0); -- 0.96522185
	signal s0_93881932 : std_logic_vector(15 downto 0); -- 0.93881932
	signal s0_89510490 : std_logic_vector(15 downto 0); -- 0.89510490
	signal s0_82709447 : std_logic_vector(15 downto 0); -- 0.82709447
	signal s0_73202614 : std_logic_vector(15 downto 0); -- 0.73202614
	signal s0_62222222 : std_logic_vector(15 downto 0); -- 0.62222222
	signal s0_53333333 : std_logic_vector(15 downto 0); -- 0.53333333
	signal s0_50000000 : std_logic_vector(15 downto 0); -- 0.50000000
	signal s0_46666667 : std_logic_vector(15 downto 0); -- 0.46666667
	signal s0_37777778 : std_logic_vector(15 downto 0); -- 0.37777778
	signal s0_26797386 : std_logic_vector(15 downto 0); -- 0.26797386
	signal s0_17290553 : std_logic_vector(15 downto 0); -- 0.17290553
	signal s0_10489510 : std_logic_vector(15 downto 0); -- 0.10489510
	signal s0_06118068 : std_logic_vector(15 downto 0); -- 0.06118068
	signal s0_03477815 : std_logic_vector(15 downto 0); -- 0.03477815

	signal in0_trunc : signed(15 downto 0); -- truncate input to Q2.13
begin
	
	in0_trunc <= in0(20 downto 5); --Q2.13

	-- assign binary constant
	s0_00772177 <= "0000000011111101"; -- 0.00772177
	s0_01526535 <= "0000000111110100"; -- 0.01526535
	s0_02983683 <= "0000001111010010"; -- 0.02983683
	s0_05704100 <= "0000011101001101"; -- 0.05704100
	s0_10457516 <= "0000110101100011"; -- 0.10457516
	s0_17777778 <= "0001011011000001"; -- 0.17777778
	s0_26666667 <= "0010001000100010"; -- 0.26666667
	s0_33333333 <= "0010101010101011"; -- 0.33333333
	
	-- x*constant
	x0_00772177 <= in0_trunc*signed(s0_00772177); -- x*0.00772177
	x0_01526535 <= in0_trunc*signed(s0_01526535); -- x*0.01526535
	x0_02983683 <= in0_trunc*signed(s0_02983683); -- x*0.02983683
	x0_05704100 <= in0_trunc*signed(s0_05704100); -- x*0.05704100
	x0_10457516 <= in0_trunc*signed(s0_10457516); -- x*0.10457516
	x0_17777778 <= in0_trunc*signed(s0_17777778); -- x*0.17777778
	x0_26666667 <= in0_trunc*signed(s0_26666667); -- x*0.26666667
	x0_33333333 <= in0_trunc*signed(s0_33333333); -- x*0.33333333

	-- assign binary constant
	s0_96522185 <= "0111101110001100"; -- 0.96522185
	s0_93881932 <= "0111100000101011"; -- 0.93881932
	s0_89510490 <= "0111001010010011"; -- 0.89510490
	s0_82709447 <= "0110100111011110"; -- 0.82709447
	s0_73202614 <= "0101110110110011"; -- 0.73202614
	s0_62222222 <= "0100111110100101"; -- 0.62222222
	s0_53333333 <= "0100010001000100"; -- 0.53333333
	s0_50000000 <= "0100000000000000"; -- 0.50000000
	s0_46666667 <= "0011101110111100"; -- 0.46666667
	s0_37777778 <= "0011000001011011"; -- 0.37777778
	s0_26797386 <= "0010001001001101"; -- 0.26797386
	s0_17290553 <= "0001011000100010"; -- 0.17290553
	s0_10489510 <= "0000110101101101"; -- 0.10489510
	s0_06118068 <= "0000011111010101"; -- 0.06118068
	s0_03477815 <= "0000010001110100"; -- 0.03477815

	process(in0, s0_03477815, s0_06118068, s0_10489510, s0_17290553,
			s0_26797386, s0_37777778, s0_46666667, s0_50000000, 
			s0_53333333, s0_62222222, s0_73202614, s0_82709447,
			s0_89510490, s0_93881932, s0_96522185,
			x0_00772177, x0_01526535, x0_02983683, x0_05704100,
			x0_10457516, x0_17777778, x0_26666667, x0_33333333)
	begin
		if (in0(31 downto 20)=x"FFF" or in0(31 downto 20)=x"000") then -- -4<x<4
			case (in0(20 downto 17)) is
				when "0111" => -- x>3.5, out=0.00772177*x+0.96522185
					out0 <= x0_00772177(28 downto 13) + signed(s0_96522185); --Q0.15
				when "0110" => -- x>3.0, out=0.01526535*x+0.93881932
					out0 <= x0_01526535(28 downto 13) + signed(s0_93881932); --Q0.15
				when "0101" => -- x>2.5, out=0.02983683*x+0.89510490
					out0 <= x0_02983683(28 downto 13) + signed(s0_89510490); --Q0.15
				when "0100" => -- x>2.0, out=0.05704100*x+0.82709447
					out0 <= x0_05704100(28 downto 13) + signed(s0_82709447); --Q0.15
				when "0011" => -- x>1.5, out=0.10457516*x+0.73202614
					out0 <= x0_10457516(28 downto 13) + signed(s0_73202614); --Q0.15
				when "0010" => -- x>1.0, out=0.17777778*x+0.62222222
					out0 <= x0_17777778(28 downto 13) + signed(s0_62222222); --Q0.15
				when "0001" => -- x>0.5, out=0.26666667*x+0.53333333
					out0 <= x0_26666667(28 downto 13) + signed(s0_53333333); --Q0.15
				when "0000" => -- x>0.0, out=0.33333333*x+0.50000000
					out0 <= x0_33333333(28 downto 13) + signed(s0_50000000); --Q0.15
				when "1111" => -- x>-0.5, out=0.33333333*x+0.50000000
					out0 <= x0_33333333(28 downto 13) + signed(s0_50000000); --Q0.15
				when "1110" => -- x>-1.0, out=0.26666667*x+0.46666667
					out0 <= x0_26666667(28 downto 13) + signed(s0_46666667); --Q0.15
				when "1101" => -- x>-1.5, out=0.17777778*x+0.37777778
					out0 <= x0_17777778(28 downto 13) + signed(s0_37777778); --Q0.15
				when "1100" => -- x>-2.0, out=0.10457516*x+0.26797386
					out0 <= x0_10457516(28 downto 13) + signed(s0_26797386); --Q0.15
				when "1011" => -- x>-2.5, out=0.05704100*x+0.17290553
					out0 <= x0_05704100(28 downto 13) + signed(s0_17290553); --Q0.15
				when "1010" => -- x>-3.0, out=0.02983683*x+0.10489510
					out0 <= x0_02983683(28 downto 13) + signed(s0_10489510); --Q0.15
				when "1001" => -- x>-3.5, out=0.01526535*x+0.06118068
					out0 <= x0_01526535(28 downto 13) + signed(s0_06118068); --Q0.15
				when "1000" => -- x>-4.0, out=0.00772177*x+0.03477815
					out0 <= x0_00772177(28 downto 13) + signed(s0_03477815); --Q0.15
				when others =>
					out0 <= (others => '0');
			end case;
		else
			if (in0(31)='0') then 	-- x>4, out=1
				out0 <= "0111111111111111";
			else					-- x<4, out=0
				out0 <= "0000000000000000";
			end if;
		end if;
	end process;
end RTL;

