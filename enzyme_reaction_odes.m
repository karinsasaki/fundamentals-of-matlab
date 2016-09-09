function dydt = enzyme_reaction_odes(T,Y,rates)

stoch = [-1 +1 0; -1 +1 1; +1 -1 -1; 0 0 +1];

substrates = [Y(1)*Y(2); Y(3); Y(3)];  

dydt = stoch*(rates.*substrates);