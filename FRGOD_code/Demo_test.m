%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1计时等声明%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic; %计时开始
clc;
clear
clear all;
format long;
%% %加载路径
% currentFolder = pwd;
% addpath(genpath(currentFolder));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%2导入数据%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load all_datalists_outlier.mat;

data_nameori=datalists{80};%11 58 62

disp(['data:' data_nameori])

eval(['load ' data_nameori ';']);% 最后一列为决策属性

Dataori=trandata;%将数据传给Data，方便后面引用

%%标准化原始数据
trandata=Dataori;%初始化

tic;
out_scores0=FRGOD_v0(trandata(:,1:end-1),1);
labels=trandata(:,end);
ROC_AUC_v0= evaluation_outlier(out_scores0,labels)
T0=toc

tic;
out_scores1=FRGOD(trandata(:,1:end-1),1);
labels=trandata(:,end);
ROC_AUC_v1= evaluation_outlier(out_scores1,labels)
T1=toc



