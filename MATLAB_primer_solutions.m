

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

% function [y] = vector_sum(x)
% 
% % This program takes in a vector x and returns a scalar y that is the sum
% % of all the entries of x
% 
% y = sum(x)
% 
% end


% 2. 

% function [max_values, values_indices] = vect_cols_max(W)
% % vect_cols_max takes in a vector v and returns the largest elements of each column
% % of v and their position in v
% 
% [sort_v, sort_indices] = sort(W);
% 
% max_values = sort_v(3,:);
% values_indices = sort_indices(3,:);


%% 4.2


% function [area, circumference, diameter] = calc_circle(radius)
% 
% area = pi*radius^2;
% 
% circumference = 2*pi*radius;
% 
% diameter = 2*radius;


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

%% 4.4


%% 6.1

% 1. 
% For the lipo-protein interactions:
xlRange = 'B2:X96';
filename = 'protein_lipid_interaction.xlsx';
lpMAT = xlsread(filename, xlRange);


% 1. 
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

figure
hold on
for i = 1:500

    % extract rows of ActinMAT for which
    % row 1 is equal to 1
    fiber_location = find(ActinMAT(:,1)==i);
    
    % create a new array that stores the
    % same information as ActinMAT, but only
    % for fiber 1
    fiber=ActinMAT(fiber_location,:);
    
    % extract the locations in |fiber1| that
    % store only point 0. Hint: use column 2
    % get the distance
    fiber_pt0_location = find(fiber(:,2)==0);
    
    % create a new array that stores the
    % same information as fiber1, but only
    % for point 0
    fiber_pt0 = fiber(fiber_pt0_location,:);
    
    % get distance
    x = fiber_pt0(:,3);
    y = fiber_pt0(:,4);
    fiber_pt0_distance = sqrt(x.^2+y.^2);
    
    % plot
    t = 1:9;
    plot(t,fiber_pt0_distance);
end


%% 8.1


%% 8.2


%close all
%clear
%clc

% import image into MATLAB
im = imread('image.png');

% im has 3 dimensions. so reduce to 1
im = im(:,:,1);

% cropp image and assing to a new variable
im_cropped = im(1:300,300:end);

% visualise the images
figure  % creates a new figure window
imshow(im_cropped)

figure % creates a new figure window
imshow(im)

% increase the contrast for better visualisation
im_eq = adapthisteq(im);

figure
imshow(im_eq)

% change image to binary (or balck and white)
im_bw = im2bw(im_eq, graythresh(im_eq));

figure
imshow(im_bw)


% "clean up"
im_bw2 = imfill(im_bw,'holes');
im_bw3 = imopen(im_bw2, ones(1,1));
im_bw4 = bwareaopen(im_bw3, 5);
figure 
imshow(im_bw4)

% cell perimeter
im_bw4_perim = bwperim(im_bw4);
figure
imshow(im_bw4_perim)

im_bw_perim = bwperim(im_bw);
figure
imshow(im_bw_perim)

% perimeter overlay
overlay1 = imoverlay(im_eq, bw4_perim, [.3 1 .3]);
figure
imshow(overlay1)

% mark a group of connected pixels inside objects that need to be segmented
im_mask_em = imextendedmax(im_eq, 15);
figure
imshow(im_mask_em)

% clean up and overlay
im_mask_em_clean = imclose(im_mask_em, ones(5,5));
im_mask_em_clean = imfill(im_mask_em, 'holes');
im_mask_em_clean = bwareaopen(im_mask_em, 40);
im_overlay2 = imoverlay(im_eq, im_bw4_perim | im_mask_em_clean, [.3 1 .3]);
imshow(im_overlay2)

% complement image (0->1 and 1->0) for watershedding
im_eq_c = imcomplement(im_eq);

im_mod = imimposemin(im_eq_c, ~im_bw4 | im_mask_em_clean);
L = watershed(im_mod);
imshow(label2rgb(L))
