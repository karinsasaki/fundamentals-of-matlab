function dydt = reversible_odes(T,Y,rates)

 stoch = [-1 1; 1 -1];
 %rates = [0.1; 0.05];
 reactants = [Y(1); Y(2)];
dydt = stoch * ((rates).*(reactants));  