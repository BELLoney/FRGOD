clc;
clear all

load Example.mat
Dataori=Example;
format shortG

trandata=Dataori;
trandata(:,2:3)=normalize(trandata(:,2:3),'range');
lam=1;

out_factors=FRGOD(trandata,lam)
