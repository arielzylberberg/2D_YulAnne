function [S,pSet] = sim2D_dw(mStr,cStr,varargin)
% [S,pSet] = sim2D(mStr,cStr) Simulates a 2D trial, using
% motion and color strengths, mStr and cStr, respectively. These will not
% correspond to strengths and performance expectations in our standard
% diffusion models. Set one of these to [] or NaN to run a 1D simulation.
%
% The model works by integrating evidence for a finite amount of
% acquisition time, tau_a (0.09s default). The buffer will be held until it
% is cleared to update the corresponding decision variable, Vm or Vc. This
% update requires tau_u (0.09s default) to begin to affect the appropriate
% V, during which time there can be no updating of the other V, but
% acquisition can resume without delay, unless the buffer is full. The
% update itself undergoes 1st order dynamics controlled by tau_V (0.05s
% default). The dimension update currenly alternates (I will add an option
% to make the alternation stochastic; not done yet) until one terminates,
% and the decision terminates when both do or t=tmax (5 s, default). 
%
% To simulate a variable duration experiment, simply set stimDur to a value
% less than tmax. For example,
%
% sim2D(mStr,cStr,'stimDur',0.12,'tmax',1)
% 
% Note that because of delays including the dynamics of V, it's wise to
% choose tmax > stimDur+0.03
%
% Return args (fields of structure S) [some of this might be out of date]
% ~~~~~~~~~~~
%       TtermM: motion termination time (nan if not terminated by tmax)
%       TtermC: color termination time
%      Tterm2D: max([TtermM,TtermC])
%     TuntermM: time that final motion update achieves
%               steadyStateFactor*tauV from final update
%     TuntermC: ibid, for color. These are useful for fixed duration
%          chC: choice 1/0 blue/yellow, based on sign of Vc(end)
%          chM: choice 1/0 right/left, based on sign of Vm(end) 
%            t: vector of time
%           dt: time stepsize
%           Vc: decision variable, color [size(t)]
%           Vm: decision variable, motion
%      VcHeavy: Vc idealized as steps
%      VmHeavy: Vm idealized as steps
%          Vrb: DV for right and blue (ignore; yet to implement properly)
%          Vry: DV for right and yello (ignore; yet to implement properly)
%          Vlb: DV for left and blue (ignore; yet to implement properly)
%          Vly: DV for left and yellow (ignore; yet to implement properly)
%     BmStatus: Status of motion buffer
%     BcStatus: Status of color buffer
%           hg: handle to Figure showing DVs as f(t)
%          hax: 1x4 vector of plot handles (if ...,'showgraph', true)
%
% 
% 
% we inform the update by integrating a portion of the evidence. This would
% be replaced by an association in the shapes task, and in general. The
% buffer contains the integral of evidence from a portion of the stimulus,
% limited to the acquisition time tau_a, and it resets it after it is read
% out.
%
% Dynamics of V is implemented as a leaky integration on step functions.
% Implying that V itelf is a perfect integrator of impulses shaped as
% exponential decay.

% 2/4/19 mns wrote it 
% 2/13 added S.Wc and W.Wn to returned structure
%      added the signal latencies to the white noise; fixed the scaling of
%      the deterministic signal (drift rate) added to white noise
% 2/18 corrected the bug in 1st part of the stimulus that made Wc and Wm
%      identical for 40 ms
% 2/19 tried to incorporate random alternation of buffers. It led to weird
%      behavior at the beginning in which both buffers were ignored. The
%      code remains in place but disabled: BUF_ALTERNATE_RAND = false
% 2/21 add decision time for shortDur: parameter steadyStateFactor and
%      return args S.TuntermC, S.TuntermM
% 2/23 adding times of buffer full status (buggy)
%      * corrected error in the signal delay adjustment to Wc and Wv
%      * counting the visLat in tau_a, but not against the duration of the stimulus
% 3/7  correcting the buffer bug (asterisk issue) but created a new problem
%      with buffer availability
% 3/9  corrected asterisks and the buffer availability. 
% 3/17 adding 1D functionality. 
% 8/11 documentation

% *****************************************
%  To do items
% *****************************************
% - proper times and buffer values when complete (for graphics)
%
% - separate bounds for color and motion
%
% -combine into target-wise DVs (so far just adding)
%
% -allow for processing post in the tnd period. Several options to consider
%   * only the last process? 
%   * adjust the heavyside to bound level but only after some delay (makes
%     it possible to place a bound on the sum.
%
% -Respect race architecture for each dimension (control anticorrelation) and incorporate in the
%  targetwise as well
% 
% -Spiking
%
% *****************************************

%% Parse varargin


pSet = inputParser;
addParameter(pSet,'tmax',5,@(x) x>0); % max time in seconds
addParameter(pSet, 'isShortT', false,@islogical)
addParameter(pSet,'stimDur',nan); 
addParameter(pSet,'BoundC', 1);
addParameter(pSet,'BoundM', 1);
addParameter(pSet,'dt',.001);
addParameter(pSet,'tau_a', 0.09, @(x) x>0); % acquisition time (buffer length)
addParameter(pSet,'tau_u', 0.09, @(x) x>0); % update time
addParameter(pSet,'tau_V',0.05,@(x) x>0); % time const of exponential impuse response for update
addParameter(pSet,'visLat',0.04,@(x) x>0); % effects only the first acquisition
addParameter(pSet,'signalLatM',0.04,@(x) x>=0); % 1st part of motion signal that is 0% coh
addParameter(pSet,'signalLatC',0,@(x) x>=0); % same for color, but default is 0
addParameter(pSet,'steadyStateFactor',3,@(x) x>=0); % multiples of tau_V to steady state (affects unterminated times) and is only used in if isShortT if isShort = true
% addParameter(pSet,'randBufferSequence',false,@islogical);

addParameter(pSet,'showGraph',false,@islogical); 


parse(pSet,varargin{:});

%% translate the pSet
% BUF_ALTERNATE_RAND = pSet.Results.randBufferSequence;
BUF_ALTERNATE_RAND = false;
v2struct(pSet.Results)

if isShortT
    if isnan(stimDur)
        error('set stimDur to a value less than tmax');
    end
else
    stimDur = tmax;
end

ONED = false;
COLOR1D = false;
MOTION1D = false;
if ~all(isfinite([cStr, mStr]))
    % 1D case
    if isfinite(cStr)
        COLOR1D = true;
    else
        MOTION1D = true;
    end
    ONED = true;
end

%% definitions and preliminaries
t = (0:dt:tmax)';
Nt = length(t);
n_u = round(tau_u/dt); % index to offset n by to respect tau_u

Wc = dt * cStr + sqrt(dt)*randn(Nt,1); % Wiener processes for color and motion
Wm = dt * mStr + sqrt(dt)*randn(Nt,1);
% The first portion of the signal might be 0% coherence
Wc(t<(visLat+signalLatC)) = Wc(t<(visLat+signalLatC)) - dt*cStr;
Wm(t<(visLat+signalLatM)) = Wm(t<(visLat+signalLatM)) - dt*mStr;
% the very first part (the visual latency) is no signal at all
Wc(t< visLat) = 0;
Wm(t< visLat) = 0;


% Implements a03_variable_dur or short-duration experiment
if stimDur<tmax
    nDur = round((stimDur+visLat)/dt); % the duration should not count the visual latency
    Wc(nDur+1:end)=0;
    Wm(nDur+1:end)=0;
end

%% Initialize

Vc = zeros(Nt,1); 
Vm = zeros(Nt,1); 
VcHeavy = zeros(Nt,1); % steps (heavyside)
VmHeavy = zeros(Nt,1); 
BcStatus = zeros(Nt,1); % binary status of the buffer
BmStatus = zeros(Nt,1); 
BcV = zeros(Nt,1); % vector of the buffer content.
BmV = zeros(Nt,1); 



% for the dynamic RDM and RDC, we inform the update by integrating a
% portion of the evidence. 
Bc = 0; % initialize the color buffer
Bm = 0; % initialize the motion buffer

% Let buffer contain the integral of evidence an reset it after it is read
% out

TQ1 = tau_a; % 1st time to query a buffer
nextUpdate = round(TQ1/dt)+1; % in steps (added +1 3/8/19)

% some mnemonics to make the code read logically
AVAILABLE = 1; 
UNAVAILABLE = 0; 
FULL = 0;   



% we keep the record of the buffer alternation. That was mainly for
% development, but it makes things easier.
BcStatus(1:end) = AVAILABLE; 
BmStatus(1:end) = AVAILABLE;
BcT = 0; % initialize the time lapsed in the buffer formation
BmT = 0;

updateStatus = UNAVAILABLE; % initialize. Can't update the DV until a buffer is ready

% State of the DVs
if MOTION1D
    Cterminated = true;
else
    Cterminated = false; % initialized
end

if COLOR1D
    Mterminated = true;
else
    Mterminated = false;
end
TtermM = nan; % Time of termination. if nan at end, there was not bound absorption
TtermC = nan;
TuntermM = nan;
TuntermC = nan;
Tunterm2D = nan;
T_BmFull = []; % times that a buffer filled
T_BcFull = [];
V_BmFull = []; % values of the full buffer
V_BcFull = [];

%% main update loop
% Which buffer to start
% Bnum = mod(unidrnd(100),2)+1; % yet to implement
Bnum = unidrnd(2);
n = 1;
DEBUG = false;
while n<Nt
    n=n+1;
    tnow = t(n);
    if DEBUG
        if tnow<=0.18
            fprintf('%d\t%.3f\t%d %d\t%d\t%.3f\t%.3f %.3f\n',...
                n,tnow,BcStatus(n),BmStatus(n),updateStatus,t(nextUpdate),BcT,BmT);
        end
        if tnow >= 0.089
            q = sprintf('debug');
        end
        % update color buffer if available
        if BcT==tau_a
            fprintf('BcT = tau_a\n');
        end
        if BmT==tau_a
            fprintf('BmT = tau_a\n');
        end
    end
    if ~MOTION1D
        if BcStatus(n) == AVAILABLE || tnow == tau_a
            % Bc = Bc + Wc(n)*dt;
            Bc = Bc + Wc(n);
            BcT = BcT+dt;
            if BcT>tau_a
                BcStatus(n+1:end) = FULL;
                % fprintf('\t B_color Full\n') % debug
                %
                % [moot comment, I think]
                %   I had these two lines commented out cuz replaced with
                %   the code at end. But it's better to do it here. What
                %   was the problem? There's some trimming that needs
                %   doing. And it's missing the 1st buffer.
                T_BcFull = [T_BcFull; tnow];
                V_BcFull = [V_BcFull; Bc];
                BcT = 0; % reset timer once full
            end
        end
        BcV(n)=Bc; % keep a record of the buffer in a vector (for plotting)
    end
    % update motion buffer if available
    if ~COLOR1D
        if BmStatus(n) == AVAILABLE || tnow==tau_a
            % Bm = Bm + Wm(n)*dt;
            Bm = Bm + Wm(n);
            BmT = BmT+dt;
            if BmT>tau_a
                BmStatus(n+1:end) = FULL;
                %   fprintf('\t B_motion Full\n')
                T_BmFull = [T_BmFull; tnow]; % see note above
                V_BmFull = [V_BmFull; Bm];
                BmT = 0; % new
            end
        end
        BmV(n) = Bm;  % keep a record of the buffer in a vector (for plotting)
    end
    % Is it possible to update the DV now?
    if n==nextUpdate % not until 1st query time t>TQ1
        updateStatus = AVAILABLE; 
    end
    
    if updateStatus == AVAILABLE
        % the code that runs here strobes an update, but it affects the Vc
        % or Vm only tau_u later, via a step, which is only realised via an
        % exponential charging
        
        % choose a buffer
        if n==2
            % Which buffer to start
            % Bnum = mod(unidrnd(100),2)+1;
            Bnum = unidrnd(2);
        end
        if 1
            % alternate deterministically
            if (mod(Bnum,2) == 1 || Mterminated) && ~Cterminated && ~MOTION1D
                % update color
                VcHeavy(n+n_u : end) =  VcHeavy(n+n_u : end) + ...
                    Bc*ones(size(VcHeavy(n+n_u : end)));
                Bc=0; % reset the buffer
                BcStatus(n+1:end) = AVAILABLE;
                BcT = 0;
            elseif ~Mterminated && ~COLOR1D
                % update motion
                VmHeavy(n+n_u : end) =  VmHeavy(n+n_u : end) +...
                    Bm*ones(size(VmHeavy(n+n_u : end)));
                Bm=0; % reset the buffer
                BmStatus(n+1:end) = AVAILABLE;
                BmT = 0;
            end
            if ~BUF_ALTERNATE_RAND  % new 2/19/19
                Bnum = Bnum+1; % alternate deterministically
            else
                Bnum = Bnum + rand<0.5; 
            end
            
        else
            % This to be filled in if I develop other ways to handle the
            % buffers. It may be unecessary since setting tau_a might give
            % the desired flexibility, and random alternation is
            % achieved simply with the BUF_ALTERNATE_RAND flag.
        end
        updateStatus = UNAVAILABLE;
        nextUpdate = n+n_u;
        if n == nextUpdate
            updateStatus = AVAILABLE;
        end
    end % of strobed update

    % the real updates
    if ~MOTION1D
        Vc(n) = updateLeakyIntegrator(Vc(n-1),VcHeavy(n),tau_V,dt);
    end
    if ~COLOR1D
        Vm(n) = updateLeakyIntegrator(Vm(n-1),VmHeavy(n),tau_V,dt);
    end
    % do the targetwise too? Maybe, but not today
    % VmRcB(n)= updateLeakyIntegrator(VmRcB(n-1),VmRcB(n),tau_V,dt);
    
    % test for bound crossings
    if ~Cterminated && abs(Vc(n)) >= BoundC
        TtermC = tnow;
        Cterminated = true;
    end
    if ~Mterminated && abs(Vm(n)) >= BoundM
        TtermM = tnow;
        Mterminated = true;
    end
end
%%  Done
if any(isnan([TtermM, TtermC]))
    % at least one porcess did not terminate
    Tterm2D = nan; % better than setting to tmax
    if pSet.Results.isShortT
        % at least one did not finish
        if isnan(TtermC) && ~MOTION1D
            IuC = find(diff(VcHeavy)~=0); % indices of updates on Vc
            tuC = t(IuC(end)) + pSet.Results.steadyStateFactor * tau_V;
            TuntermC = tuC;
        else
            TuntermC = nan; % maybe make this nan
        end
        if isnan(TtermM) && ~COLOR1D
            IuM = find(diff(VmHeavy)~=0); % indices of updates on Vm
            tuM = t(IuM(end)) + pSet.Results.steadyStateFactor * tau_V;
            TuntermM = tuM;
        else
            TuntermM = nan;
        end 
        Tunterm2D = max([TuntermM TuntermC]);
    end
else
    Tterm2D = max([TtermM TtermC]);
    Tunterm2D = nan;
end




S.chC = sign(Vc(end));
S.chM = sign(Vm(end));
S.t = t;
S.dt = dt;
S.stimDur = stimDur;
S.Wm = Wm; % added for use in Fig
S.Wc = Wc;
S.BmStatus = BmStatus; % status of the buffers
S.BcStatus = BcStatus;
S.BcV = BcV; % record of buffer content

% % When did the buffer fill and what was the value
% I = find(diff(S.BcStatus)==-1 & S.t(1:end-1)<=stimDur);
% J = find(t==pSet.Results.stimDur+visLat | t==tau_a);
% for j = 1:length(J)
%     if BcStatus(J(j))==AVAILABLE
%         I = [I;J(j)];
%     end
% end
% S.T_BcFull = t(I); % times buffer
% S.V_BcFull = S.BcV(I);

% fix the time of buffer fill. Kludge to correct that I miss the first
%      buffer full of the dimension that is read out first (3/8/19)
if ~any(T_BcFull == tau_a)
    T_BcFull = [tau_a; T_BcFull];
    V_BcFull = [BcV(t==tau_a); V_BcFull];
end
if ~any(T_BmFull == tau_a)
    T_BmFull = [tau_a; T_BmFull];
    V_BmFull = [BmV(t==tau_a); V_BmFull];
end
% remove buffer fills after stimDur
if any(T_BcFull > stimDur + visLat) || any(T_BcFull > TtermC)
    tStop = min([stimDur + visLat, TtermC]);
    L = T_BcFull > tStop;
    T_BcFull(L) = [];
    V_BcFull(L) = [];
end
if any(T_BmFull > stimDur + visLat) || any(T_BmFull > TtermM)
    tStop = min([stimDur + visLat, TtermM]);
    L = T_BmFull > tStop;
    T_BmFull(L) = [];
    V_BmFull(L) = [];
end


S.T_BcFull = T_BcFull;
S.V_BcFull = V_BcFull;

S.BmV = BmV;
% % I'll move this. It finds the times that the buffer is full and available
% I = find(diff(S.BmStatus)==-1 & S.t(1:end-1)<=stimDur);
% J = find(t==pSet.Results.stimDur+visLat | t==tau_a);
% for j = 1:length(J)
%     if BmStatus(J(j))==AVAILABLE
%         I = [I;J(j)];
%     end
% end
% I = unique(I);
% S.T_BmFull = t(I);
% S.V_BmFull = S.BmV(I);
S.T_BmFull = T_BmFull;
S.V_BmFull = V_BmFull;

S.Vc = Vc;
S.Vm = Vm;
S.VcHeavy = VcHeavy;
S.VmHeavy = VmHeavy;


S.Vrb = Vm+Vc;
S.Vry = Vm-Vc;
S.Vlb = -S.Vry;
S.Vly = -S.Vrb;
S.TtermM = TtermM;
S.TtermC = TtermC;
S.Tterm2D = Tterm2D;
S.TuntermM = TuntermM;
S.TuntermC = TuntermC;
S.Tunterm2D = Tunterm2D;



if pSet.Results.showGraph
    % Look at buffer alternation
    figure, clf, hold on
    title('Buffers')
    plot(t,BmStatus,'k-','linewidth',2)
    plot(t,BcStatus,'c--','linewidth',2)
    plot(t,0*t,'k-')
    xlabel('Time (s)');
    ylabel('Acquisition on/off');
    
    %
    cgrey = 0.6*[1 1 1];
    S.hfig = figure;clf, hold on
    S.hg(1) = plot(t,Vc,'b-');
    S.hg(2) = plot(t,Vm,'k-');
    S.hg(3) = plot(t,VcHeavy,'m--');
    S.hg(4) = plot(t,VmHeavy,'--','color',cgrey);
    xlabel('Time (s)');
    ylabel('Decision variable');
end

