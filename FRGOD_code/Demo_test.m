%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1��ʱ������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic; %��ʱ��ʼ
clc;
clear
clear all;
format long;
%% %����·��
% currentFolder = pwd;
% addpath(genpath(currentFolder));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%2��������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load all_datalists_outlier.mat;

data_nameori=datalists{80};%11 58 62

disp(['data:' data_nameori])

eval(['load ' data_nameori ';']);% ���һ��Ϊ��������

Dataori=trandata;%�����ݴ���Data�������������

%%��׼��ԭʼ����
trandata=Dataori;%��ʼ��

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



