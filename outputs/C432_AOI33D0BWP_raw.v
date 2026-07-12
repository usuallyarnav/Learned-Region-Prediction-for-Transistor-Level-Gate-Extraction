module c432(inout N1, inout N4, inout N8, inout N11, inout N14, inout N17, inout N21, inout N24, inout N27, inout N30, inout N34, inout N37, inout N40, inout N43, inout N47, inout N50, inout N53, inout N56, inout N60, inout N63, inout N66, inout N69, inout N73, inout N76, inout N79, inout N82, inout N86, inout N89, inout N92, inout N95, inout N99, inout N102, inout N105, inout N108, inout N112, inout N115, inout N223, inout N329, inout N370, inout N421, inout N430, inout N431, inout N432, inout VDD, inout VSS);
	AOI33D0BWP X1 (.A1(n164), .A2(n127), .A3(n165), .B1(n134), .B2(n133), .B3(n166), .ZN(n137), .VDD(VDD), .VSS(VSS));
	// MATCH AOI33D0BWP M308 M307 M306 M305 M304 M303 M302 M301 M300 M299 M298 M297
	AOI33D0BWP X2 (.A1(n156), .A2(n160), .A3(N4), .B1(n155), .B2(n174), .B3(N30), .ZN(n169), .VDD(VDD), .VSS(VSS));
	// MATCH AOI33D0BWP M422 M421 M420 M419 M418 M417 M416 M415 M414 M413 M412 M411
	AOI33D0BWP X3 (.A1(n156), .A2(n157), .A3(n108), .B1(n118), .B2(n117), .B3(n158), .ZN(n139), .VDD(VDD), .VSS(VSS));
	// MATCH AOI33D0BWP M260 M259 M258 M257 M256 M255 M254 M253 M252 M251 M250 M249
endmodule

// Runtime:   0.072913 s
// Instances: 3
// Coverage:  36/520 (6.92%)
