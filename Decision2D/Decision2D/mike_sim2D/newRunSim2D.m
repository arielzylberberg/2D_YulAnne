% newRunSim2D.m  

%% preliminaries
clear all
rng('shuffle');
% deleteAllFigures
orange = [1 157/255 0];
%  myParPool  (for cluster; ignore)

%% test the function


mStr = 1 % pos means evidence for right
cStr = nan  % pos means evidence for blue
ntr = 1;
[S,pSet] = sim2D(mStr,cStr,'showGraph',true); % 1D motion

%% A short-duration trial
mStr = 1
cStr = 4
[S,pSet] = sim2D(mStr,cStr,'isShortT',true,'tmax',.8,'stimDur',0.2,'showGraph',true,...
    'BoundC',.4,'BoundM',.4);

[S.TtermM S.TtermC]


figure(S.hfig)
% delete(S.hg(3:4))
set(S.hg(1:2),'linewidth',2)
hl = legend('Vc','Vm')

%% Short dur
ntr = 2;
% dbstop if error
strengths = 2*sort([2.^(1:-.5:-3)';0])
[M,C] = meshgrid(strengths,strengths)
nstr = length(strengths);
chM = nan(ntr,nstr,nstr);
chC = nan(ntr,nstr,nstr);
DT = nan(ntr,nstr,nstr);

cStr = 0; 
stimeStart = datestr(now)
for k = 1:nstr
    for j = 1:nstr
        parfor i = 1:ntr
            % [S,pSet] = sim2D(M(j,k),C(j,k));
            [S,pSet] = sim2D(M(j,k),C(j,k),'isShortT',true,'tmax',1.2,'stimDur',0.8);
            chM(i,j,k) = S.chM;
            chC(i,j,k) = S.chC;
            if any(isnan([S.TtermM,S.TtermC]))
                DT(i,j,k) = S.Tunterm2D;
            else
                DT(i,j,k) = S.Tterm2D;
            end
        end
    end
end
stimeEnd = datestr(now)

[S,pSet] = sim2D(M(1,1),C(1,1),'isShortT',true,'tmax',1.2,'stimDur',0.8);

[y,m,d,h,mi,s] = datevec(now) 
matFileName = sprintf('sim2Dout_%d_%d_%d_%d_%d',y,m,d,h,mi)
s = sprintf('save %s DT M C ch* stime* ntr nstr pSet str*',matFileName)
eval(s)

%% Variable duration  ** THIS WORKS **
% unfold the coherences. This almost doubles the number of trials
ntr = 2; % n per coh. 
ndur = ntr;
boundHeight = 0.3;

% random durations 
durMin = 0.1;
durMax = 1.2;
stimDur = linspace(durMin,durMax,ntr); % thes will be indexed by i in the loop.

% dbstop if error
strengthsC = 2*sort([2.^(1:-.5:-3)'; -2.^(1:-.5:-3)'; 0]);
strengthsM = 2*sort([strengthsC(1);0]);
[T,M,C] = ndgrid(stimDur,strengthsM,strengthsC);
nstrM = length(strengthsM);
nstrC = length(strengthsC);
chM = nan(ntr,nstrM,nstrC);
chC = nan(ntr,nstrM,nstrC);
DT = nan(ntr,nstrM,nstrC);
termC = nan(ntr,nstrM,nstrC);
termM = nan(ntr,nstrM,nstrC);

cStr = 0; 
ONED = true;
stimeStart = datestr(now)
for k = 1:nstrC
    for j = 1:nstrM
        parfor i = 1:ntr
        % for i = 1:ntr
            % [S,pSet] = sim2D(M(j,k),C(j,k));
            % [S,pSet] = sim2D(M(j,k),C(j,k),'isShortT',true,'tmax',1.2,'stimDur',0.8); 
            if ONED == true
                [S,pSet] = sim2D(nan,C(i,j,k),'isShortT',true,'stimDur',stimDur(i),...
                    'tmax',stimDur(i)+0.4, 'BoundC',boundHeight,'BoundM',boundHeight);
            else
                [S,pSet] = sim2D(M(i,j,k),C(i,j,k),'isShortT',true,'stimDur',stimDur(i),...
                'tmax',stimDur(i)+0.4, 'BoundC',boundHeight,'BoundM',boundHeight);
            end
            chM(i,j,k) = S.chM;
            chC(i,j,k) = S.chC;
            termC(i,j,k) = S.TtermC;
            termM(i,j,k) = S.TtermM;
            if any(isnan([S.TtermM,S.TtermC]))
                DT(i,j,k) = S.Tunterm2D;
            else
                DT(i,j,k) = S.Tterm2D;
            end
        end
    end
end
stimeEnd = datestr(now)

[S,pSet] = sim2D(M(1,1,1),C(1,1,1),'isShortT',true,'tmax',0.4,'stimDur',0.2,...
    'BoundC',boundHeight,'BoundM',boundHeight);

[y,m,d,h,mi,s] = datevec(now) 
S.readme = {'Variable duration. 1D Color only. stimDur indexed by array T. Motion subsampled'}
matFileName = sprintf('sim2Dout_%d_%d_%d_%d_%d',y,m,d,h,mi)
s = sprintf('save %s S DT M C T ch* stime* ntr nstr* pSet str* stimDur term*',matFileName)
eval(s)

%% Look at the simulation
% load sim2Dout_2019_2_10_21_12.mat
% load sim2Dout_2019_3_8_15_44.mat
% load sim2Dout_2019_3_18_13_15.mat % test of shortT
% load sim2Dout_2019_3_22_7_0 % 1D color dur 0.6
% load sim2Dout_2019_3_22_7_6 % 2D dur 0.6 5 trials
% eval(['load ' matFileName]) % load the latest matFileName
% load sim2Dout_2019_3_22_7_55.mat % 2D dur = 0.8 s (100 trials per)
load sim2Dout_2019_8_12_9_0.mat
unique(chC(:))

chC(chC==-1) = 0;
chM(chM==-1) = 0;


DTm = squeeze(mean(DT,1));
chMm = squeeze(mean(chM,1));
chCm = squeeze(mean(chC,1));




pSet.Results
if any(isnan(DT(:)))    
    DT5 = DT;
    DT5(isnan(DT))=pSet.Results.tmax + 3*pSet.Results.tau_V; % set to max
    size(DT5)
    
    DT5m = squeeze(mean(DT5,1))
end

% for 2D case
size(DTm)
L_M0 = M==0; 
L_Mmax = abs(M)==max(M(:));
Pcor = [chCm(L_M0) chCm(L_Mmax)];
Pcor(C(L_M0)==0,:) = .5
% Pcor = (1+[chCm(L_M0) chCm(L_Mmax)])/2

hFigPMF = figure; hold on;
[b0,dev0] = glmfit(C(L_M0), Pcor(:,1),'binomial','link','logit');
[bmx,devmx] = glmfit(C(L_Mmax), Pcor(:,2),'binomial','link','logit');
x = linspace(0,4);
yfit0 = glmval(b0,x,'logit');
yfitmx = glmval(bmx,x,'logit');
plot(x,yfit0,'k--');
plot(x,yfitmx,'k-');
plot(C(L_M0),Pcor(:,1),'ko','markerfacecolor','w'); 
plot(C(L_Mmax),Pcor(:,2),'ko','markerfacecolor','k')
set(gca,'xscale','log')
set(gca,'xlim',[.95*C(2), 1.05 * C(end)])


chCm_M0
figure(1),clf, hold on
plot(strengths,(1+mean(chMm,1))/2,'o-')
xlabel('Motion strength')
figure(2),clf, hold on
plot(strengths,(1+mean(chCm,2))/2,'o-')
xlabel('Color strength')

% figure(3), clf, hold on
% for i = 1:nstr
%     plot(strengths,DT5m(i,:),'-o')
% end
% legend(num2str(strengths))
% 
% pSet.Results.tmax

