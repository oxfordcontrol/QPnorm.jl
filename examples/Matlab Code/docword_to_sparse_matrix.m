M = dlmread('../docword.nytimes.txt', ' ', 3, 0);
D = spconvert(M);
clear M;
save docword_nytimes.mat
