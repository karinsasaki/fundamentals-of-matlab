% MATLAB Fundamentals
% EMBL-Heidelberg
% 27/01/2015
% Debugg Example
% Automate the creation of a matrix with specific values:
% the value of entry (r,c) is 1/(r+c)

function create_funky_array(s)
%H = zeros(s);

for r = 1:1:s
    for c = 1:1:s
        H(r,c) = [H 1/(r+c);
    end
end