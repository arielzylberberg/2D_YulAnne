% script for simulations and analysis in Hyafil and Moreno-Bote, eLife 2017
% "Breaking down hierarchies of decision-making in primates"
% replicate Lorteije et al 2015 Neuron paper with a flat race model (LCA of Usher
% 2001)
%
% Address any comment to alexandre.hyafil (at) gmail.com

clear; close all;

newset = 0; % 1 for new set of parameters (to accomodate for RTs in Zylberberg et al)
propnoise = 0; % whether noise is proportional to signal
withinh = 1; % whether we include inhibition
noisetype = 'option'; % 'stim' if noise is added at the level of stimulus, 'option' it is at the level of option

k2 = .012; % weight for L2 evidence
k1 = .016; % weight for L1 evidence
Idc = .5;   % constant input
x0 = .5; % initial point of activation
noneg = 1; % whether activations are rectified (i.e. no negative value)

if ~withinh, newset = 0; end

if newset % changes to accomodate for RTs
    nsamp_min = 10; % minumum number of samples before decision can be formed
    threshold =  30; % decision threshold
    alpha = .05*withinh; % autoexcitation
    lambda = -.12*withinh; % %!!! inhibition ('beta' in manuscript)
else % set that produces large RTs
    nsamp_min = 0; % minumum number of samples before decision can be formed
    if withinh
        threshold =  50; % decision threshold
    else
        threshold = 6;
    end
    alpha = .1*withinh; % autoexcitation
    lambda = -.07*withinh; % %!!! inhibition ('beta' in manuscript)
end

% noise parameters
if strcmp(noisetype, 'option') % noise added at level of option ('accumulation noise')
    if propnoise, % if noise is proportional to signal
        eta = .3; 
        eta_prop = 1.5;
    else
        eta = 1; % variance of white noise
        eta_prop = 0;
    end
else % noise added at level of stimulus ('sensory noise')
    eta = 40;
    eta_prop = 0;
end

%%%%

%% generate stimuli
ntrial = 100000;  % total number of trials
nsamp =  20; % number of samples per trials (50 ms each, 500 ms overall)
nsamp_real = 10;
lum_target  = [45 31.5 25]; % average luminosity of target path for easy, medium and difficult trials (for medium and difficult, we take the mean of the two used values)
lum_distractor = 15; % average luminositry of distractor path
lum_std = 10; % std of luminosity

difficulty = randsample(1:3, ntrial,1); % level of difficulty of each trial: easy, medium of difficult

target = randsample(1:4,ntrial,1); % correct answer for each trial
target1 = ceil(target/2); % correct decision at first level
target2 = mod(target-1,2)+1; % correct decision at second level
target2p = randsample(1:2,ntrial,1); % 'correct' decision at L2'
target_secondbranch = zeros(2,ntrial); % which is the correct decision for each of two second branches
target_secondbranch(:,target1==1) = [target2(target1==1); target2p(target1==1)]; % when first secondary branch is L2, second is L2'
target_secondbranch(:,target1==2) = [target2p(target1==2); target2(target1==2)]; % when first secondary branch is L2, second is L2'

% index of TT, TD, DT and DD
iTT = 2*(target1-1) + target2;
iTD = 2*(target1-1) + 3-target2;
iDT = 2*(2-target1)+ target2p;
iDD = 2*(2-target1) + 3-target2p;
allis = [iTT; iTD; iDT; iDD];  % 4 x ntrial matrix

% build matrix of samples luminosity (and response to trial type mapping)
luminosity = zeros(6,ntrial,nsamp); %
luminosity2_stimtype = zeros(4,ntrial,nsamp); % luminosty at L2 branches for TT, TD, DT and DD
resptype = zeros(4,ntrial); % 1:TT, 2:TD, 3:DT, 4:DD
resptype_label = {'TT','TD','DT','DD'};
for t=1:ntrial
    this_lum_target = lum_target(difficulty(t)); % average luminosity of target paths for this trial
    
    % response to trial mapping
    resptype(allis(:,t),t) = [1 2 3 4];
    
    % in L1
    luminosity(target1(t),t,:) = normrnd(this_lum_target,lum_std,[1 1 nsamp]);  % target L1
    luminosity(3-target1(t),t,:) = normrnd(lum_distractor,lum_std,[1 1 nsamp]);  % distractor L1
    
    % in L2
    luminosity2_stimtype(1,t,:) = normrnd(this_lum_target,lum_std,[1 1  nsamp]);  % target L2 (TT)
    luminosity2_stimtype(2,t,:) = normrnd(lum_distractor,lum_std,[1 1 nsamp]);  % distractor L2 (TD)
    
    % in L2'
    luminosity2_stimtype(3,t,:) = normrnd(this_lum_target,lum_std,[1 1 nsamp]);  % target L2' (DT)
    luminosity2_stimtype(4,t,:) = normrnd(lum_distractor,lum_std,[1 1 nsamp]);  % distractor L2' (DD)
    
    %
    luminosity(2+allis(:,t),t,:) =  luminosity2_stimtype(:,t,:);  
end

%% generate behaviour

% matrix of connectivity from individual sample luminositry to differential
% sample luminosity for each 3 branches
samp2diff = zeros(6,3);
samp2diff([1 2],1) = [1 -1]; % L1
samp2diff([3 4],2) = [1 -1]; % first secondary branch
samp2diff([5 6],3) = [1 -1]; % second secondary branch

% matrix of connectivity from differential luminositty to representation of
% responses
diff2resp = [k1  k2  0 ;
    k1  -k2 0  ;
    -k1 0   k2 ;
    -k1 0   -k2]';

% matrix of connectivity from sample luminositty to representation of
% responses
samp2resp = samp2diff * diff2resp; % 6x4 matrix

% initialize value
X = zeros(4,ntrial,nsamp+1); % activity of each response representation
X(:,:,1) = x0;

responded = false(1,ntrial); % whether one representation has reached threshold for response
resp = zeros(1,ntrial);
resp1 = zeros(1,ntrial);
resp_samp = zeros(1,ntrial); % at which sample decision is reached
resptype_trial = zeros(1,ntrial); % whether response at each trial is TT,TD, DT or DD
accu_L1 = zeros(1,ntrial); % accuracy of L1 decision
accu_L2 = zeros(1,ntrial); % accuracy of L2/L2' decision


% accumulate over samples
for s =1:nsamp
    update =  true(1,ntrial);  % whether to add evidence also for trials with already a response
       
    % new evidence in favor of each of 4 options
    newevidence = samp2resp' * luminosity(:,update,s);
    
    % noise
    if strcmp(noisetype, 'option'), % noise directly at the level of options
        newnoise =  (eta + eta_prop*newevidence ).*normrnd(0,1,4,sum(update));
    else % noise at the level of stimulus evidence
        % !! warning : if propnoise is not null, might give wierd results
        newnoise =  (eta + eta_prop*newevidence ).*(samp2resp' * normrnd(0,1,6,sum(update)));
    end
    
    newevidence = newnoise + newevidence;    
    
    % add to previously accumulated evidence (do not accumulate if response
    % has already been reached)
    newX = bsxfun(@minus, (1 - lambda+ alpha)*X(:,update,s), alpha*sum(X(:,update,s),1)); %
    newX = newX + newevidence + Idc;
    
    %enforce non-negative values
    if noneg,
        newX = max(newX,0);
    end
    
    X(:,update,s+1) = newX;
    
    % for already responded no more evolution
    X(:,~update,s+1) = X(:,~update,s);
    
    % trials where a new response if found
    if s>= nsamp_min, % can only respond if has reached fixation off
    i_newresp = find(~responded & any(X(:,:,s+1) >= threshold));
    responded(i_newresp) = true;
    resp_samp(i_newresp) = s;
    for t=i_newresp,
        [~, resp(t)] = max(X(:,t,s+1)); % what is the response
        resp1(t) = ceil(resp(t)/2); % response at first branche
        resptype_trial(t) = resptype(resp(t),t); % whether it corresponds to TT,TD,DT or DD
        accu_L1(t) = any(resptype_trial(t)==[1 2]);  % accuracy at first branch
        accu_L2(t) = any(resptype_trial(t)==[1 3]); % accuracy at second branch
    end
    end
end

% trials where response has not been reached after all samples are
% presented : just take maximal value for response
resp_samp(~responded) = nsamp+1;
for t=find(~responded),
    maxt = max(X(:,t,nsamp+1)); % what is the response
    maxX = find(X(:,t,nsamp+1)==maxt); %  maximum activation
    if length(maxX)>1,
        resp(t) = randsample(maxX,1);  %if more than one, pick random)
    else
        resp(t) = maxX;
    end
    resp1(t) = ceil(resp(t)/2); % response at first branche
    resptype_trial(t) = resptype(resp(t),t); % whether it corresponds to TT,TD,DT or DD}
    accu_L1(t) = any(resptype_trial(t)==[1 2]);
    accu_L2(t) = any(resptype_trial(t)==[1 3]);
end

resp2 = mod(resp-1,2)+1; % response at second branch

%% analyse behaviour : set parameters to match response type distrib, sample weights and psychometric curves

% luminosity of T1, D1, T2, and D2
luminosity_T1 = zeros(nsamp,ntrial);
luminosity_D1 = zeros(nsamp,ntrial);
luminosity_T2 = zeros(nsamp,ntrial);
luminosity_D2 = zeros(nsamp,ntrial);
diff_luminosity1 = zeros(nsamp,ntrial);
diff_luminosity2 = zeros(nsamp,ntrial); % used to compute psychometric function

diff_luminosity2_L1 = zeros(nsamp,ntrial); % differential luminosity for L2 left branch
diff_luminosity2_R1 = zeros(nsamp,ntrial); % differential luminosity for L2 right branch


for t=1:ntrial
    luminosity_T1(:,t) = luminosity(target1(t),t,:);
    luminosity_D1(:,t) = luminosity(3-target1(t),t,:);
    
    luminosity_T2(:,t) = luminosity(2+ 2*(resp1(t)-1) + target_secondbranch(resp1(t),t)  ,t,:); %luminosity target at secondary branch (depending on which branche was chosen at L1)
    luminosity_D2(:,t) = luminosity(2+ 2*(resp1(t)-1)+3-target_secondbranch(resp1(t),t)  ,t,:); %luminosity target at secondary branch (depending on which branche was chosen at L1)
    
    diff_luminosity1(:,t) = luminosity(2,t,:)-luminosity(1,t,:); % difference of luminosity between left and right options at L1
    diff_luminosity2(:,t) = luminosity(2*(resp1(t)-1)+4,t,:)-luminosity(2*(resp1(t)-1)+3,t,:); % difference of luminosity between left and right options at L2 (in chosen L1 path)
    
    diff_luminosity2_L1(:,t) = luminosity(4,t,:)-luminosity(3,t,:); % difference of luminosity between left and right options at L2 left branch
    diff_luminosity2_R1(:,t) = luminosity(6,t,:)-luminosity(5,t,:); % difference of luminosity between left and right options at L2 right branch
    
end
luminosity1 = [luminosity_T1;luminosity_D1];
luminosity2 = [luminosity_T2;luminosity_D2];

%ease of L2 relative to L2' (for each sample) : (TT-TD)-(DT-DD)
Fdiff = squeeze(luminosity2_stimtype(1,:,:) - luminosity2_stimtype(1,:,:))' - squeeze(luminosity2_stimtype(3,:,:) - luminosity2_stimtype(4,:,:))';


% selection signal for L2 and L2'
X_stimtype = zeros(4,ntrial,nsamp+1); % activity of each response representation, sorted with 1:TT, 2:TD, 3:DT, 4:DD
for t=1:ntrial
    X_stimtype(allis(:,t),t,:) = X(:,t,:);
end
Xdiff2 = squeeze( X_stimtype(1,:,:) - X_stimtype(2,:,:) ); % selection signal at L2 (TT-TD)
Xdiff2prime = squeeze( X_stimtype(3,:,:) - X_stimtype(4,:,:) ); % selection signal at L2' (DT-DD)
Xdiff2sel(resp1==1,:) = squeeze(diff(X(1:2,resp1==1,:))); %selection signal in the selected L2 branch:
Xdiff2sel(resp1==2,:) = squeeze(diff(X(3:4,resp1==2,:))); %selection signal in the selected L2 branch:
Xmean2sel(resp1==1,:) = squeeze(mean(X(1:2,resp1==1,:))); %mean signal in the selected L2 branch:
Xmean2sel(resp1==2,:) = squeeze(mean(X(3:4,resp1==2,:))); %mean signal in the selected L2 branch:
%luminosity_diff2sel(resp1==1,:) = squeeze(diff(luminosity(3:4,resp1==1,:))); %differential luminosity in the selected L2 branch:
%luminosity_diff2sel(resp1==2,:) = squeeze(diff(luminosity(5:6,resp1==2,:))); %differential luminosity in the selected L2 branch:

Xd2(target1==1,:)= squeeze(diff(X(1:2,target1==1,:)));
Xd2(target1==2,:)= squeeze(diff(X(3:4,target1==2,:)));
Xd2prime(target1==1,:)= squeeze(diff(X(3:4,target1==1,:)));
Xd2prime(target1==2,:)= squeeze(diff(X(1:2,target1==2,:)));
luminosity_d2(target1==1,:) = squeeze(diff(luminosity(3:4,target1==1,:))); %differential luminosity in the  L2 branch:
luminosity_d2(target1==2,:) = squeeze(diff(luminosity(5:6,target1==2,:))); %differential luminosity in the  L2 branch:
luminosity_d2prime(target1==1,:) = squeeze(diff(luminosity(5:6,target1==1,:))); %differential luminosity in the  L2' branch:
luminosity_d2prime(target1==2,:) = squeeze(diff(luminosity(3:4,target1==2,:))); %differential luminosity in the ' L2' branch:

%% type of response per difficulty
figure; set(gcf, 'name','response type'); hold on;
resptype_diff = zeros(3,4);
for i=1:3
    for j=1:4
        resptype_diff(i,j) = sum( difficulty==i & resptype_trial==j); % count number of trial type for each level of difficulty
    end
end
resptype_diff = bsxfun(@rdivide, resptype_diff, sum(resptype_diff,2));
colors = [0 108 45; 84 140 36; 168 185 28; 255 230 19]/255;
  bb =   bar(resptype_diff); 
  for i=1:4, set(bb(i), 'facecolor',colors(i,:)); end
set(gca, 'xtick',1:3,'xticklabel', {'easy','interm','difficult'});
legend(resptype_label);
  

%% influence of luminosities at each time step on L1 decision
figure; set(gcf, 'name','luminosity weights L1'); hold on;
baselum = lum_target(difficulty); % base luminance of target (variable L associated with beta4 in original paper)
XX_L1 = [luminosity1; Fdiff; baselum]';
[beta_L1, ~, S_L1] = glmfit(XX_L1, accu_L1', 'binomial');
bias1 = beta_L1(1);
beta_T1 = beta_L1(2:nsamp+1);
beta_T1_ste = S_L1.se(2:nsamp+1);
beta_D1 = beta_L1(nsamp+(2:nsamp+1));
beta_D1_ste = S_L1.se(nsamp+(2:nsamp+1));
beta_Fdiff = beta_L1(2*nsamp+(2:nsamp+1));
beta_Fdiff_ste = S_L1.se(2*nsamp+(2:nsamp+1));
betanormfact = sum(beta_T1);
wu2(beta_T1/betanormfact, beta_T1_ste/betanormfact ,'color',[222 125 0]/255, 'errorstyle','fill');
wu2(beta_D1/betanormfact, beta_D1_ste/betanormfact ,'color',[126 47 142]/255, 'errorstyle','fill');
plot(1:nsamp,zeros(1,nsamp),'k--');
xlabel('sample'); ylabel('beta (norm.)');

%% influence of luminosites at each time step on L2 decision
figure; set(gcf, 'name','luminosity weights L2'); hold on;
XX_L2 = [luminosity2; baselum]';
[beta_L2, ~, S_L2] = glmfit(XX_L2, accu_L2', 'binomial');
bias2 = beta_L2(1);
beta_T2 = beta_L2(2:nsamp+1);
beta_T2_ste = S_L2.se(2:nsamp+1);
beta_D2 = beta_L2(nsamp+(2:nsamp+1));
beta_D2_ste = S_L2.se(nsamp+(2:nsamp+1));
betanormfact2 =sum(beta_T2);
wu2(beta_T2 ./betanormfact2 , beta_T2_ste/betanormfact2, 'color',[222 125 0]/255, 'errorstyle','fill');
wu2(beta_D2 ./ betanormfact2,beta_D2_ste/betanormfact2, 'color',[126 47 142]/255, 'errorstyle','fill');
plot(1:nsamp,zeros(1,nsamp),'k--');
xlabel('sample'); ylabel('beta (norm.)');

%% psychometric curve
figure; set(gcf, 'name','psychometric'); hold on;
% psychometric curve for L1
XX_psycho1 = [luminosity_T1 - luminosity_D1; baselum]';
[w_L1, ~, S_L1] = glmfit(XX_L1, accu_L1', 'binomial');
w_L1_samp = w_L1(2:nsamp+1) / sum(w_L1(2:nsamp+1));
ev1 = w_L1_samp(1:nsamp_real)' *diff_luminosity1(1:nsamp_real,:); % weighted  evidence at L1

ev_edges = -50:1:50;
n_ev_edges = length(ev_edges);
[~, ~, ev1_hist] = histcounts(ev1, ev_edges );
resp_ev1 = zeros(1,n_ev_edges);
resp_ev1_se = zeros(1,n_ev_edges);
for u=1:n_ev_edges
   this_trial =  (ev1_hist == u); %select trials from this bin
   if any(this_trial)
   resp_ev1(u) = mean(resp1(this_trial))-1; % proportion of right response
   resp_ev1_se(u) = std(resp1(this_trial)) / sqrt(sum(this_trial)); % standard error
   end
end
fill([ev_edges fliplr(ev_edges)], [resp_ev1+resp_ev1_se fliplr(resp_ev1-resp_ev1_se)], [1 .5 .5], 'linestyle', 'none');
plot(ev_edges, resp_ev1, 'color',[1 0 0]);

% same for L2
XX_psycho2 = [luminosity_T2 - luminosity_D2; baselum]';
[w_L2, ~, S_L2] = glmfit(XX_L2, accu_L2', 'binomial');
w_L2_samp = w_L2(2:nsamp+1) / sum(w_L2(2:nsamp+1));
ev2 = w_L2_samp(1:nsamp_real)' *diff_luminosity2(1:nsamp_real,:);

[~, ~, ev2_hist] = histcounts(ev2, ev_edges );
resp_ev2 = zeros(1,n_ev_edges);
resp_ev2_se = zeros(1,n_ev_edges);
for u=1:n_ev_edges
   this_trial =  (ev2_hist == u); %select trials from this bin
   if any(this_trial)
   resp_ev2(u) = mean(resp2(this_trial))-1; % proportion of right response
   resp_ev2_se(u) = std(resp2(this_trial)) / sqrt(sum(this_trial)); % standard error
   end
end
fill([ev_edges fliplr(ev_edges)], [resp_ev2+resp_ev2_se fliplr(resp_ev2-resp_ev2_se)], .5+.5*[0 204 205]/255, 'linestyle', 'none');
plot(ev_edges, resp_ev2, 'color',[0 204 205]/255);
xlim([-40 40]);
xlabel('evidence at L1/L2 (cd/m^2)');
ylabel('p-rightward choices');

%% response sample (i.e. reaction times minus non-decision processes)
% mean reaction times
figure; set(gcf, 'name','reaction times'); hold on;
for d=1:3
   this_trial = difficulty  ==d;
   resp_samp_diff(d) = mean(resp_samp(this_trial));
      resp_samp_diff_se(d) = std(resp_samp(this_trial))/sqrt(length(this_trial));
end
bar(50/1000*resp_samp_diff, 'facecolor',.8*[1 1 1]);
errorbar(1:3, 50/1000*resp_samp_diff, 50/1000*resp_samp_diff_se, 'color','k', 'linestyle','none');
set(gca,'xtick',1:3, 'xticklabel', {'E' 'I' 'D'});
ylabel('reaction time (s)'); ylim([0 1.5]); xlim([-1 5]);

L1cols = [247 147 29; 237 52 147]/255; % colors for l1 T and D
L2cols = [5 148 70; 230 228 31;  159 75 156; 74 165 220]/255; %% colors for L2: TT, tD, DT, DD
sumLcols = (L1cols([1 1 2 2],:) + L2cols)/2; % 


%% example trial
figure; set(gcf, 'name','activation example'); hold on;
this_nsamp = 15; % find a trial where response is after 15 samples
itrial = find(accu_L1 & accu_L2 & resp_samp==this_nsamp, 1);
thisX = permute(X_stimtype(:,itrial,1:this_nsamp+1), [3 1 2]); % activation for this trial
for u=1:4
   plot( 0:this_nsamp, thisX(:,u),'color', sumLcols(u,:));
end
legend(resptype_label);
uu = this_nsamp-1+(threshold-thisX(end-1,1))/(thisX(end,1)-thisX(end-1,1)); % at which X exactly passes the boundary
fill([uu uu this_nsamp this_nsamp  uu], [ylim fliplr(ylim) min(ylim)], 1*[1 1 1],'linestyle', 'none'); %  % mask the part beyond threshold crossing
plot(xlim, [0 0], 'color', .8*[1 1 1]); 
plot([nsamp_min this_nsamp], threshold*[1 1], 'color', .2*[1 1 1]);
xlabel('sample'); ylabel('activation'); ylim([0 threshold+2]);


%% %%% test 4 critical behavioural and activations predictions %%%%

%% behavioural : influence of L1 difficulty on L2 (psychometric curve)
figure; set(gcf, 'name','L2 accuracy x L1 difficulty'); hold on;
n_evbin = 40;
[ev2_bin, ev_bin_edges] = quantiles(ev2, n_evbin);
ev_histc_sgn = sign(ev_bin_edges);
ev1_diff = quantiles(abs(ev1), 2); % split trials into easy and hard L1
colors = [50 27 113; 0 204 255]/255;
   for j=1:2
       for i=1:n_evbin
       this_trial = ev2_bin==i & ev1_diff ==j; % trials with given level of L2 evidence and L1 difficulty
       psycho2_diff1(i,j) = mean(resp2(this_trial))-1;
              psycho2_diff1_se(i,j) = std(resp2(this_trial))/sqrt(sum(this_trial));
       end
   fill([ev_bin_edges fliplr(ev_bin_edges)], [psycho2_diff1(:,j)+psycho2_diff1_se(:,j);flipud(psycho2_diff1(:,j)-psycho2_diff1_se(:,j))], .5+.5*colors(j,:), 'linestyle','none');
   hh(j) = plot(ev_bin_edges, psycho2_diff1(:,j), 'color',colors(j,:));
end
legend(hh, {'low L1','high L1'});
psycho2_diff1_effect = nansum(ev_histc_sgn' .* diff(psycho2_diff1,1,2));
ylabel('p-rightward at L2');
xlim([-40 40]);

%% compute stats
n_perm =  500;
alldiff_perm = zeros(1,n_perm);
for i_perm = 1:n_perm
    for bb = 1:n_evbin % shuffle L1 difficulty labels independently in each L2 evidence bin (to keep sample number bias)
        this_evbin = find(ev2_bin==bb); % trials within a L2 evidence bin
        perm_trial = this_evbin(randperm(length(this_evbin))); % shuffle those trials
        this_L1diff =  ev1_diff(perm_trial);
        ev1_rnd( this_evbin) = this_L1diff; % re-assign L1-difficulty
        
         % compute split psychometric curve for this shuffled data
        for j=1:2
            psycho2_rnd(bb,j) = mean(resp2(this_evbin(this_L1diff==j)))-1;
        end
    end
      
    alldiff_perm(i_perm) = nansum(ev_histc_sgn' .* diff(psycho2_rnd,1,2));
end
quant_perm = mean(alldiff_perm < psycho2_diff1_effect);
fprintf('comparing with random permutation of L1 difficulty, quantile %f, %d perms\n',quant_perm, n_perm);



%% behavioural : influence of L2 and L2' difficulties on L1
figure; set(gcf, 'name','L1 choice x L2 difficulty'); hold on;
ev2_L1 =  w_L2_samp' *diff_luminosity2_L1; % evidence at L2 left branch
ev2_R1 =  w_L2_samp' *diff_luminosity2_R1; % evidence at L2 right branch
diff_difficulty = abs(ev2_R1) - abs(ev2_L1); % difference of difficulty between two branches
subplot(4,1,1:3); hold on;
[diff_easyhard, ffquants] = quantiles(diff_difficulty, [0 .25 .75]); % split trials into easy and hard L1
%!!!
ev1_hist2_bins = -50:2:50;
[~,~,ev1_hist2 ] = histcounts(ev1, ev1_hist2_bins); % replace L1 evidence by quantile+
colors = [0 .8 0; 1 .2 .2];
    for j=1:2 % easy/hard trials
for i=1:length(ev1_hist2_bins)-1
        this_trial = ev1_hist2==i & diff_easyhard==1+2*(j>1); % trials with given L1 evidence 
    accuL2_L1diff(i,j) = mean(resp1(this_trial))-1;
        accuL2_L1diff_se(i,j) = std(resp1(this_trial))/sqrt(sum(this_trial));
end
    fill([ev1_hist2_bins(1:end-1) fliplr(ev1_hist2_bins(1:end-1))], [accuL2_L1diff(:,j)+accuL2_L1diff_se(:,j);flipud(accuL2_L1diff(:,j)-accuL2_L1diff_se(:,j))],.5+.5*colors(j,:), 'linestyle','none');
    hh(j) = plot(ev1_hist2_bins(1:end-1), accuL2_L1diff(:,j), 'color', colors(j,:));
    
end

xx_accuL2_L1diffdiff = ev1_hist2_bins(1:end-1);
yy_accuL2_L1diffdiff = diff(accuL2_L1diff,1,2)';
hfill = fill([xx_accuL2_L1diffdiff(~isnan(accuL2_L1diff(:,1)')) fliplr(xx_accuL2_L1diffdiff(~isnan(accuL2_L1diff(:,2)')))],...
    [accuL2_L1diff(~isnan(accuL2_L1diff(:,1)),1)' fliplr(accuL2_L1diff(~isnan(accuL2_L1diff(:,2)),2)')], [.8 .8 .8],'linestyle','none') ;
uistack(hfill,'bottom');
xlim([-40 40]); set(gca, 'xtick', []);
ylabel('p-right');
subplot(414);
xx_accuL2_L1diffdiff(isnan(yy_accuL2_L1diffdiff)) = [];
yy_accuL2_L1diffdiff(isnan(yy_accuL2_L1diffdiff)) = [];
fill([xx_accuL2_L1diffdiff xx_accuL2_L1diffdiff(end:-1:1)],[yy_accuL2_L1diffdiff 0*yy_accuL2_L1diffdiff], [.8 .8 .8]  );
box off; axis tight;
xlim([-40 40]); xlabel('evidence L1 (cd/m2)');

%% plot histogram of difference in L2 difficulty
figure; set(gcf, 'name','L2 difficulty difference'); hold on;
edges1 = ffquants(2)+(-diff(ffquants(1:2)):.5:0);
edges2 = ffquants(2)+(0:.5:diff(ffquants(2:3)));
edges3 = ffquants(3)+(0:.5:max(diff_difficulty)-ffquants(3));
hist_difdif1 = histc(diff_difficulty(diff_easyhard==1), edges1);
hist_difdif2 = histc(diff_difficulty(diff_easyhard==2), edges2);
hist_difdif3 = histc(diff_difficulty(diff_easyhard==3), edges3);
h_hist1 = bar(edges1, hist_difdif1, 'histc');
h_hist2 = bar(edges2, hist_difdif2, 'histc');
h_hist3 = bar(edges3, hist_difdif3, 'histc');
set(h_hist1, 'facecolor',[0 .8 0], 'linestyle','none');
set(h_hist2, 'facecolor','k', 'linestyle','none');
set(h_hist3, 'facecolor',[1 .2  .2], 'linestyle','none');
xlim([-25 25]);

%% plot Influence of L2 onto L1
figure; set(gcf, 'name','Influence L2 on L1'); hold on;
wu2(beta_Fdiff/betanormfact, beta_Fdiff_ste/betanormfact ,'color',[0 0 0], 'errorstyle','fill');
plot(zeros(1,nsamp),'k')
xlabel('sample');
ylabel('beta');
ylim([-max(ylim)/2 max(ylim)]);

%% neural : profile of L2 and L2' differential activity
normfactorXdi = min(threshold, 10);
figure; set(gcf, 'name','L2 selection signals');
subplot(4,1,1:3);hold on;
Xdiff2_norm = Xdiff2/normfactorXdi; % normalize activation
xdif_mean= mean(Xdiff2_norm,1);
xdif_ste = std(Xdiff2_norm,[],1)/sqrt(ntrial);
fill([0:nsamp nsamp:-1:0], [xdif_mean+xdif_ste fliplr(xdif_mean-xdif_ste)], .5+.5*[0 204 255]/255, 'linestyle','none');
plot(0:nsamp,xdif_mean, 'color',[0 204 255]/255);

Xdiff2prime_norm = Xdiff2prime/normfactorXdi; % normalize activation
xdifprime_mean= mean(Xdiff2prime_norm,1);
xdifprime_ste = std(Xdiff2prime_norm,[],1)/sqrt(ntrial);
fill([0:nsamp nsamp:-1:0], [xdifprime_mean+xdifprime_ste fliplr(xdifprime_mean-xdifprime_ste)], .5+.5*[50 27 113]/255, 'linestyle','none');
plot(0:nsamp,xdifprime_mean, 'color',[50 27 113]/255);

hfill = fill([0:nsamp nsamp:-1:0]', [xdif_mean' ;flipud(xdifprime_mean')], .8*[1 1 1], 'linestyle','none');
uistack(hfill,'bottom');
ylabel('norm modulation');
subplot(4,1,4); hold on;
xdif_dif = xdif_mean - xdifprime_mean;
fill([0:nsamp nsamp:-1:0]', [xdif_dif' ;zeros(nsamp+1,1)], .8*[1 1 1], 'linestyle','none');
plot(0:nsamp, xdif_dif, 'color', [0 108 45]/255, 'linewidth', 2);
xlabel('sample');


%% neural : reciprocal influence of L2/L2' difficulty onto the other
% differential activity
figure; set(gcf, 'name','L2'' influence on L2'); hold on;
Xdiff2prime_q = quantiles(abs(Xdiff2prime(:,2))',2); % low vs high L2' evidence trials
colors = [0 51 204;0 92 24]/255;
for j=1:2
Xdiff2_ev2prime(:,j) = mean(  Xdiff2_norm(Xdiff2prime_q==j,:));
Xdiff2_ev2prime_ste(:,j) = std(  Xdiff2_norm(Xdiff2prime_q==j,:)) / sqrt(sum(Xdiff2prime_q==j));
fill([0:nsamp nsamp:-1:0], [Xdiff2_ev2prime(:,j)+Xdiff2_ev2prime_ste(:,j); flipud(Xdiff2_ev2prime(:,j)-Xdiff2_ev2prime_ste(:,j))], .5+.5*colors(j,:), 'linestyle','none');
hh(j) =plot(0:nsamp,Xdiff2_ev2prime(:,j), 'color',colors(j,:));
end
legend(hh, {'low L2''','high L2'''});
xlabel('sample'); ylabel('TT-TD');
xlim([0 nsamp]);

%%
figure; set(gcf, 'name','L2 influence on L2'''); hold on;
Xdiff2_q = quantiles(abs(Xdiff2(:,2))',2); % low vs high L2' evidence trials
colors = [0 51 204;0 92 24]/255;
for j=1:2
Xdiff2prime_ev2(:,j) = mean(  Xdiff2prime_norm(Xdiff2_q==j,:));
Xdiff2prime_ev2_ste(:,j) = std(  Xdiff2prime_norm(Xdiff2_q==j,:)) / sqrt(sum(Xdiff2_q==j));
fill([0:nsamp nsamp:-1:0], [Xdiff2prime_ev2(:,j)+Xdiff2prime_ev2_ste(:,j); flipud(Xdiff2prime_ev2(:,j)-Xdiff2prime_ev2_ste(:,j))], .5+.5*colors(j,:), 'linestyle','none');
hh(j) = plot(0:nsamp,Xdiff2prime_ev2(:,j), 'color',colors(j,:));
end
legend(hh, {'low L2','high L2'});
xlabel('sample'); ylabel('DT-DD');
xlim([0 nsamp]);

%% neural: modulation of all 4 signals by L1 evidence
figure; set(gcf, 'name','selection signal x L1 evidence'); hold on;
which_trial = find(accu_L1 & accu_L2);  % select correct trials
ev1_diff_quart = quantiles(abs(ev1(which_trial)), 4); % split trials into quartiles based on L1 evidence
colors = {[0 0 1],[0 .8 0],[.8 .8 0],[222 125 0]/255};
for ii = 1:4
    subplot(1,5,ii); hold on;
    for j=1:4
        this_trial =which_trial(ev1_diff_quart==j);
     X_stim_ev1(:,j,ii) = mean(X_stimtype(ii,this_trial,:),2);
          X_stim_ev1_se(:,j,ii) = std(X_stimtype(ii,this_trial,:),[],2) / sqrt(sum(this_trial));
          fill([0:nsamp nsamp:-1:0], [X_stim_ev1(:,j,ii)+X_stim_ev1_se(:,j,ii); flipud(X_stim_ev1(:,j,ii)-X_stim_ev1_se(:,j,ii))],...
              .5+.5*colors{j}, 'linestyle','none');
        hh(j) =plot(0:nsamp,X_stim_ev1(:,j,ii), 'color',colors{j});

    end
    
    title(resptype_label{ii});
end
subplot(1,5,5); hold on;

for ll=1:4
   [rr_L1corr(ll), pp_L1corr(ll)] = corr(mean( X_stimtype(ll,which_trial,:),3)', abs(ev1(which_trial))', 'type', 'spearman');
    hh = bar(ll,abs(rr_L1corr(ll)));
    set(hh, 'facecolor', sumLcols(ll,:));
end
 ylabel('abs(rho)');
set(gca, 'xtick', 1:4, 'xticklabel', resptype_label);
title('correlation');
