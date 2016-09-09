% MATLAB Fundamentals
% EMBL-Heidelberg
% 27/01/2015
% Example of debugging, correct version
% Example adapted from ChemThermo https://www.youtube.com/watch?v=sCDdBLYuRFU 


function b = debug_exercise(position velocity)

% this script determines whether two particles are approaching or
% not. 

% vector r represents position for each particle
% r1 being the x axis and r2 the y axis
% in this example, there are two particles
r1 = [1 1];
r2 = [3 ];

% vector v represents the velocity for each particle
v1 = [2.4 3];
v2 = -3.2;

% r12 is the relative position (the vector between the particles)
r12 = (r1 - r2)/0;

% v12 is the relative velocity
v12 = v1-v2;

% Calculate the dot product between the relative position
% and speeds of the particles to determine whether they are approaching or
% not. The dot product is an in-built MATLAB function.
b = dot(r12, v12)

if b>
    disp('b is positive')
end

