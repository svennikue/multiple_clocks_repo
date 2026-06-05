addpath /home/fs0/xpsy1114/scratch/analysis/fmt

% GLM ('regression') settings (creating the 'bins'):
%     01 - instruction EVs
%     02 - 80 regressors; every task is divided into 4 rewards + 4 paths
%     03 - 40 regressors; for every tasks, only the rewards are modelled [using a stick function]
%     04 - 40 regressors; for every task, only the paths are modelled


x_one = read_vest('/home/fs0/xpsy1114/scratch/data/derivatives/sub-03/func/glm_01_pt01.feat/design.mat');
x_two = read_vest('/home/fs0/xpsy1114/scratch/data/derivatives/sub-03/func/glm_02_pt01.feat/design.mat');
x_three = read_vest('/home/fs0/xpsy1114/scratch/data/derivatives/sub-03/func/glm_03_pt01.feat/design.mat');
x_four = read_vest('/home/fs0/xpsy1114/scratch/data/derivatives/sub-03/func/glm_04_pt01.feat/design.mat');


figure;plot(x_one(:,1:11));
figure;imagesc(corr(x_one));
figure;imagesc(corr(x_one(:,1:11)));

figure;plot(x_two(:,1:81));
figure;imagesc(corr(x_two));
figure;imagesc(corr(x_two(:,1:81)));

figure;plot(x_three(:,1:41));
figure;imagesc(corr(x_three));
figure;imagesc(corr(x_three(:,1:41)));

figure;plot(x_four(:,1:41));
figure;imagesc(corr(x_four));
figure;imagesc(corr(x_four(:,1:41)));

