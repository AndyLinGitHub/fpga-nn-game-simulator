----------------------------------------------------------------------------------
-- This is a design to calculate multiply and MAC

-- the input will be 16 input & 16 weight (each is 16 bit signed Q2.13)
-- the output will be the 16 multiplication and 1 MAC (each is 32 bit signed Q13.18)

-- design by Tzu-Chi Huang
----------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mac4x4 is
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
end mac4x4;

architecture RTL of mac4x4 is
	type  array_16x16  is array (15 downto 0) of
                      signed(15 downto 0);
	type  array_16x32  is array (15 downto 0) of
                      signed(31 downto 0);
    type  array_16x36  is array (15 downto 0) of
                      signed(35 downto 0);

	signal mul : array_16x32;
	signal mul_ext : array_16x32;
begin

	mul(0)  <= in0*wgt0; --Q2.13*Q2.13=Q5.26
	mul(1)  <= in1*wgt1; --Q2.13*Q2.13=Q5.26
	mul(2)  <= in2*wgt2; --Q2.13*Q2.13=Q5.26
	mul(3)  <= in3*wgt3; --Q2.13*Q2.13=Q5.26
	mul(4)  <= in4*wgt4; --Q2.13*Q2.13=Q5.26
	mul(5)  <= in5*wgt5; --Q2.13*Q2.13=Q5.26
	mul(6)  <= in6*wgt6; --Q2.13*Q2.13=Q5.26
	mul(7)  <= in7*wgt7; --Q2.13*Q2.13=Q5.26
	mul(8)  <= in8*wgt8; --Q2.13*Q2.13=Q5.26
	mul(9)  <= in9*wgt9; --Q2.13*Q2.13=Q5.26
	mul(10) <= in10*wgt10; --Q2.13*Q2.13=Q5.26
	mul(11) <= in11*wgt11; --Q2.13*Q2.13=Q5.26
	mul(12) <= in12*wgt12; --Q2.13*Q2.13=Q5.26
	mul(13) <= in13*wgt13; --Q2.13*Q2.13=Q5.26
	mul(14) <= in14*wgt14; --Q2.13*Q2.13=Q5.26
	mul(15) <= in15*wgt15; --Q2.13*Q2.13=Q5.26

	--transfer Q5.26 to Q13.18
	mul_ext(0) <= (7 downto 0 => mul(0)(31))&mul(0)(31 downto 8);
	mul_ext(1) <= (7 downto 0 => mul(1)(31))&mul(1)(31 downto 8);
	mul_ext(2) <= (7 downto 0 => mul(2)(31))&mul(2)(31 downto 8);
	mul_ext(3) <= (7 downto 0 => mul(3)(31))&mul(3)(31 downto 8);
	mul_ext(4) <= (7 downto 0 => mul(4)(31))&mul(4)(31 downto 8);
	mul_ext(5) <= (7 downto 0 => mul(5)(31))&mul(5)(31 downto 8);
	mul_ext(6) <= (7 downto 0 => mul(6)(31))&mul(6)(31 downto 8);
	mul_ext(7) <= (7 downto 0 => mul(7)(31))&mul(7)(31 downto 8);
	mul_ext(8) <= (7 downto 0 => mul(8)(31))&mul(8)(31 downto 8);
	mul_ext(9) <= (7 downto 0 => mul(9)(31))&mul(9)(31 downto 8);
	mul_ext(10) <= (7 downto 0 => mul(10)(31))&mul(10)(31 downto 8);
	mul_ext(11) <= (7 downto 0 => mul(11)(31))&mul(11)(31 downto 8);
	mul_ext(12) <= (7 downto 0 => mul(12)(31))&mul(12)(31 downto 8);
	mul_ext(13) <= (7 downto 0 => mul(13)(31))&mul(13)(31 downto 8);
	mul_ext(14) <= (7 downto 0 => mul(14)(31))&mul(14)(31 downto 8);
	mul_ext(15) <= (7 downto 0 => mul(15)(31))&mul(15)(31 downto 8);

	mul0 <= mul_ext(0);
	mul1 <= mul_ext(1);
	mul2 <= mul_ext(2);
	mul3 <= mul_ext(3);
	mul4 <= mul_ext(4);
	mul5 <= mul_ext(5);
	mul6 <= mul_ext(6);
	mul7 <= mul_ext(7);
	mul8 <= mul_ext(8);
	mul9 <= mul_ext(9);
	mul10 <= mul_ext(10);
	mul11 <= mul_ext(11);
	mul12 <= mul_ext(12);
	mul13 <= mul_ext(13);
	mul14 <= mul_ext(14);
	mul15 <= mul_ext(15);

	-- MAC = sum of 16 multiplication
	MAC <= mul_ext( 0)+mul_ext( 1)+mul_ext( 2)+mul_ext( 3)+
		   mul_ext( 4)+mul_ext( 5)+mul_ext( 6)+mul_ext( 7)+
		   mul_ext( 8)+mul_ext( 9)+mul_ext(10)+mul_ext(11)+
		   mul_ext(12)+mul_ext(13)+mul_ext(14)+mul_ext(15);
				
	
end RTL;

