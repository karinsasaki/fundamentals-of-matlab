

%% 2.1
%1. 

%2. 
diary command_window_2012015

% 3.
k = 111-345;
l = 3456*pi
m = log(cos(0.1))

% 4. 
save workspace_27012015

% 5.
clc

%% 2.2

% 1.
A = randi(10, [3,3])
a = randi(10, [3,1])
b = randi(10, [3,1])

% 2.
c = 5*a

% 3. 
a'*b        % a 1x1 array
a*b'        % a 3x3 array
a.*b        % a 3x1 arra

% 4.
% Entry in row 1 column 2 of matrix A.
% Column 3 of A.
% The intersection of rows 1 and 2 with columns 1 and 2.

% 5. 
A(:,2)=b

% 6.
A(2, 1)
A(3, :)
A(2:3,2:3)

% 7. 
A*b
A % 3 x 1 column vector. 
% No because they need to be the same dimension.

% 8. 
B = [b b b]

% 9.
C = A.*B

% 10. 
[dimx, dimy] = size(C)

% 11.
% help length
% length is equivalent to max(size(C))

% 12. 
C(1,:) = []

% 13. 
% C is a 2 x 3 matrix

% 14.
find(C<5)
[x, y] = find(C<5)


%% 2.3

% 1.

name = 'Karin';

% 2.

sentence = 'My name is ';

% 3.

my_name = [sentence name];

% 4.

K = randi(3);
L = zeros(3);
M = K==L;

% 5.
molecules(1).name = 'Kat1';
molecules(1).weight = 127.00;
molecules(1).test = [79, 75, 73; 180, 178, 177.5; 220, 210, 205];
molecules(2).name = 'Kat2';
molecules(2).weight = 130.00;
molecules(2).test = [75, 85, 73; 170, 278, 174.5; 280, 210, 225];
molecules

% 6.

whos

%% 2.4


% 1. 
Q = rand(24,3)

% 2.
mn = min(Q)             % Find the minimum value in each column
mx = max(Q)             % Find the maximum value in each column
mu = mean(Q)            % Calculate the mean of each column
sigma = std(Q)          % Calculate the standard deviation of each column

% 3.
[mx,indx] = max(Q)


% 4.
Q = rand(24,3)
[r,c] = size(Q)         % Get the size of the count matrix
mu = mean(Q)            % Compute the mean of each column
mean_matrix = repmat(mu,r,1)% Create a matrix of mean values by
                        % replicating the mu vector for n rows
star = Q - mean_matrix     % Subtract the column mean from each element
                        % in that column

%% 2.5

% 1/2.
tic
x = rand(1,1000);
x_sum = sum(x.^2);
toc

%% 4.1

% 1. 

function [y] = vector_sum(x)

% This program takes in a vector x and returns a scalar y that is the sum
% of all the entries of x

y = sum(x)

end


% 2. 

function [max_values, values_indices] = vect_cols_max(W)
% vect_cols_max takes in a vector v and returns the largest elements of each column
% of v and their position in v

[sort_v, sort_indices] = sort(W);

max_values = sort_v(3,:);
values_indices = sort_indices(3,:);


%% 4.2


function [area, circumference, diameter] = calc_circle(radius)

area = pi*radius^2;

circumference = 2*pi*radius;

diameter = 2*radius;


%% 4.3

% 1. (i)
% n = 7   gives m = 8
% n = 9   gives m = -1
% n = -10 gives m = -11
%       
%       
% % 1. (ii)
% x = -1  gives y = -4
% x = 5   gives y = 20
% x = 30  gives y = 120
% x = 100 gives y = 400

% 3. 

A = rand(4,7);
[M,N] = size(A);
for j = 1:M
    for k = 1:N
        if A(j,k) < 0.2
            A(j,k) = 0;
        else
            A(j,k) = 1;
        end
	end
end


%% 6.1

% 2. 
% For the lipo-protein interactions:
xlRange = 'B2:X96';
filename = 'protein_lipid_interaction.xlsx';
lpMAT = xlsread(filename, xlRange);


% Learn this from http://uk.mathworks.com/help/matlab/import_export/supported-file-formats.html

load('./actin.mat')
load('./lipids.mat')

%% 6.2

%--------------------
% lipid data
%--------------------

% heat map
figure(1), imagesc(lpMAT)

% compare single lipid-protein interaction with coupled lipid-protein
% interaction
figure(2), stem(x(1:12), lpMAT(:,1:12)')
figure(3), stem(x(1:11), lpMAT(:,13:23)')

% edit the plots and add labels, legend and tile using the plotting tools

% Observations 
% - it seems lipid cooperation increases protein recruitment.
% - there seem to be some lipids that inhibit each other


%--------------------
% contractile network
%--------------------


function [actin_filaments] = actin_trajectories(actinMAT)

actin_filaments(500).id = [];
actin_filaments(500).point = 0;
actin_filaments(500).coordinates = [];
actin_filaments(500).distance = [];

figure
hold on 
for i = 1:500
    % pick point 0 for every frame for filament 1
    filament1_row_ids = find(actinMAT(:,2)==0 & actinMAT(:,1)==i);
    filament1 = actinMAT(filament1_row_ids,:);
    dist = sqrt(filament1(:,3).^2 + filament1(:,4).^2);
    
    % save info
    actin_filaments(i).id = i;
    actin_filaments(i).point = 0;
    actin_filaments(i).coordinates = filament1(:,3:4);
    actin_filaments(i).distance = dist;
    
    % define time
    time = 1:length(filament1_row_ids);

    % plot
    plot(time, dist)
end
hold off


%% 8.1
% -------------------------
% Decay
% -------------------------

    % -------------------------
    % function
    % -------------------------
 
function dAdt = decay_odes(t, A, k) 

dAdt = -k*A; 
    
    % -------------------------
    % script
    % -------------------------
    
% initial concentration
A0 = 100;

% time of simulation
tspan = [0:1:100];

% reation constant
k = 0.01;

% solve system of ODEs describing decay A -> B
[Tode, Yode] = ode45(@decay_odes, tspan, A0,[],k); 

figure
hold on
plot (Tode, Yode, '--')    
xlabel('Time');
ylabel('Concentration');


% -------------------------
% Equlibrium
% -------------------------

    % -------------------------
    % function
    % -------------------------
    
function dydt = reversible_odes(T,Y,rates)

stoch = [-1 1; 1 -1];
substrates = [Y(1); Y(2)];
dydt = stoch * ((rates).*(substrates));    
    
    % -------------------------
    % script
    % -------------------------
    
Y0 = [1 0];

t = 0:100;

rates = [0.1; 0.05];

[Tode Yode] = ode45(@reversible_odes, t, Y0, [], rates); 

%plot solution of reversible system
figure, plot (Tode, Yode); 
legend ('[A]', '[B]'); 
xlabel('Time');
ylabel('Concentration');
%savefig('reversible_reaction_ode');

% -------------------------
% Enzymatic reaction
% -------------------------

    % -------------------------
    % function
    % -------------------------
% ode for 
% E + S <-> ES with rate kf and kr
% ES -> P + E with rate k

function dydt = enzyme_reaction_odes(T,Y,rates)

stoch = [-1 +1 0; -1 +1 1; +1 -1 -1; 0 0 +1];

substrates = [Y(1)*Y(2); Y(3); Y(3)];  

dydt = stoch*(rates.*substrates);
    
    
    % -------------------------
    % script
    % -------------------------

Y0 = [20 10 0 0];

t = 0:100;

rates = [0.01; 0.001; 0.01];

[Tode Yode] = ode45(@enzyme_reaction_odes, t, Y0, [], rates); 

%plot solution of reversible system
figure, plot (Tode, Yode); 
legend ('[S]', '[E]', '[SE]', '[P]'); 
xlabel('Time');
ylabel('Concentration');


% -------------------------
% Gene expression
% -------------------------

    % -------------------------
    % function
    % -------------------------
    
    
    % -------------------------
    % script
    % -------------------------

% -------------------------
% Gene regulation
% -------------------------

    % -------------------------
    % function
    % -------------------------
    
    
    % -------------------------
    % script
    % -------------------------

% -------------------------
% Challenges
% -------------------------

    % -------------------------
    % function
    % -------------------------
    
    
    % -------------------------
    % script
    % -------------------------

%% 8.2

%close all
%clear
%clc

% import image into MATLAB
im = imread('cells.png');

% im has 3 dimensions. so reduce to 1 channel
im = im(:,:,1);

% cropp image and assing to a new variable
%im_cropped = im(1:300,300:end);

% visualise the images
%figure  % creates a new figure window
%imshow(im_cropped)

figure(1) % creates a new figure window
imshow(im)

%------------------------------------------------------------
% modifications on im_eq (image with adjusted contrast)

% increase the contrast for better visualisation
im_eq = adapthisteq(im);

figure(2)
imshow(im_eq)

% mark a group of connected pixels inside objects that need to be segmented
im_mask_em = imextendedmax(im_eq, 15);

figure(3)
imshow(im_mask_em)

% clean up and overlay
im_mask_em_clean = imclose(im_mask_em, ones(5,5));
im_mask_em_clean = imfill(im_mask_em, 'holes');
im_mask_em_clean = bwareaopen(im_mask_em, 40);
figure(4)
imshow(im_mask_em_clean)
%im_overlay2 = imoverlay(im_eq, im_bw4_perim | im_mask_em_clean, [.3 1 .3]);
%imshow(im_overlay2)




%------------------------------------------------------------
% modifications on im_bw4 (black and white image)

% change image to binary (or balck and white)
im_bw = im2bw(im_eq, graythresh(im_eq));

figure(5)
imshow(im_bw)


% "clean up"
im_bw2 = imfill(im_bw,'holes');
im_bw3 = imopen(im_bw2, ones(1,1));
im_bw4 = bwareaopen(im_bw3, 5);
figure(6)
imshow(im_bw4)

%------------------------------------------------------------
% modifications on im_bw4_perim (black and white image of the perimeters of cells)

% cell perimeter
im_bw4_perim = bwperim(im_bw4);
figure(7)
imshow(im_bw4_perim)


%------------------------------------------------------------
% watershed

% complement original, cropped, contrast adjusted image (0->1 and 1->0) for watershedding
im_eq_c = imcomplement(im_eq);
figure(8)
imshow(im_eq_c)

im_mod = imimposemin(im_eq_c, ~im_bw4 | im_mask_em_clean);

% visualise to facilitate understanding of this step
figure(9)
imshow(im_bw4)
figure(10)
imshow(~im_bw4)
figure(11)
imshow(im_mask_em_clean)
figure(12)
imshow(im_bw4 | im_mask_em_clean)
figure(13)
imshow(im_mod)


L = watershed(im_mod);
figure(14)
imshow(L)
figure(15)
imshow(label2rgb(L))
