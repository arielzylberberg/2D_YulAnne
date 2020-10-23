classdef SimDtbD1 ...
        < IxnKernel.EvTime.CommonWorkspace
    % IxnKernel.SimDtb.SimDtbD1 - Simulates evidence and DTB numerically
    
%% Settings
properties
    drift = 0; % scalar
    sigmasq_ev = 1/2; % scalar
    sigmasq_internal = 1/2; % (tr, fr). Expanded if necessary.
    
    bound = [-1, 1]; % [lb, ub]
    
    n_trial = 2500; % 1e4; % scalar
    
    % smooth_ev_args
    % {'gamma_ms', mean_ms, std_ms}
    % {'normal', std_ms}
    smooth_ev_args = {'normal', 0};

    % smooth_internal_args
    % {'gamma_ms', mean_ms, std_ms}
    % {'normal', std_ms}
    smooth_internal_args = {'normal', 0};
    
    to_set_nan_aft_rt = true; % Set ev to NaN from RT, as in RT experiment
end
%% Internal
properties
    init_done = false;
end
properties (Dependent)
    noise_internal % (tr, t)
    ev % (tr, t)
    ch % (tr, 1)
    
    cev % (tr, t) % Cumulative evidence
end
properties (Transient)
    cev_ % (tr, t)
end
%% Results
properties
    % Ev = IxnKernel.EvTime.EvTimeD1; % Inherited from IxnKernel.EvTime.CommonWorkspace
end
%% Init
methods
    function Sim = SimDtbD1(varargin)
        if nargin > 0
            Sim.init(varargin{:});
        end
    end
    function init(Sim, varargin)
        varargin2props(Sim, varargin, true);
       
        Sim.copy_file_fields2Ev;
        Sim.init_done = true;
    end
    function copy_file_fields2Ev(Sim)
        Sim.Ev.copy_file_fields(Sim);
    end
end
%% Batch
methods
    function batch(Sim0, varargin)
        S_batch = varargin2S(varargin, {
            'smooth_ev_args', {
                {'normal', 0}
                {'normal', 50}
                {'normal', 100}
                }
            });
        [Ss, n] = factorizeS(S_batch);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            Sim = feval(class(Sim0), C{:});
            Sim.main;
        end
    end
end
%% Main
methods
    function main(Sim)
        if ~Sim.init_done
            Sim.init;
        end
        Sim.simulate;
        Sim.plot_and_save_all;
    end
    function simulate(Sim)
        Sim.get_ev;
        Sim.get_noise_internal;
        Sim.get_ch_rt;
        Sim.set_nan_aft_rt;
    end
    function ev = get_ev(Sim, varargin)
        S = varargin2S(varargin, {
            'n_trial', Sim.n_trial;
            'nt', Sim.nt;            
            });
        
        mu = Sim.drift .* Sim.dt;
        sig_ev = sqrt(Sim.sigmasq_ev .* Sim.dt);
        if isscalar(mu) && isscalar(sig_ev)
            ev = normrnd(mu, sig_ev, [S.n_trial, S.nt]);
        else
            mu = rep2fit(mu, [S.n_trial, S.nt], ...
                'assert_multiple', true);
            sig_ev = rep2fit(sig_ev, [S.n_trial, S.nt], ...
                'assert_multiple', true);
            ev = normrnd(mu, sig_ev);
        end
        
        if ~isempty(Sim.smooth_ev_args) %  > 0
            ev = Sim.smooth_mat(ev, Sim.smooth_ev_args);
%             ev = smooth_gauss( ...
%                 ev', ...
%                 Sim.smooth_ev_args / 1e3 / Sim.dt)';
        end
        
        if nargout == 0 % DEBUG: Commented out once
            Sim.Ev.ev = ev;
            Sim.cev = [];
        end
    end
    function noise_internal = get_noise_internal(Sim, varargin)
        S = varargin2S(varargin, {
            'n_trial', Sim.n_trial;
            'nt', Sim.nt;            
            });
                
        sig_internal = sqrt(Sim.sigmasq_internal * Sim.dt);
        noise_internal = normrnd(0, sig_internal, [S.n_trial, S.nt]);
        if ~isempty(Sim.smooth_internal_args) %  > 0
            noise_internal = Sim.smooth_mat( ...
                noise_internal, ...
                Sim.smooth_internal_args);
%             noise_internal = smooth_gauss( ...
%                 noise_internal', ...
%                 Sim.smooth_internal_args / 1e3 / Sim.dt)';
        end
        
        if nargout == 0
            Sim.Ev.noise_internal = noise_internal;
            Sim.cev = [];
        end
    end
    function mat = smooth_mat(Sim, mat, args)
        % mat(tr, fr)
        %
        % args:
        % {'gamma_ms', mean_ms, std_ms}
        % {'normal', std_ms}
        
        switch args{1}
            case 'gamma_ms'
                t = Sim.t(:);
                p = gampdf_ms(t,  ...
                    args{2}/1e3, args{3}/1e3, 1);
                mat = conv_t(mat', p)';
                
            case 'normal'
                mat = smooth_gauss( ...
                    mat', ...
                    args{2} / 1e3 / Sim.dt)';
                
        end
    end
    function get_ch_rt(Sim)
        n_trial = Sim.n_trial;
        nt = Sim.nt;
        
        cev = Sim.cev;
        
        crossed_lb = cev < Sim.bound(1);
        crossed_ub = cev > Sim.bound(2);
        
        ch = false(n_trial, 1);
        td = zeros(n_trial, 1);
        
        for i_trial = 1:n_trial
            lb_1st = find(crossed_lb(i_trial, :), 1, 'first');
            ub_1st = find(crossed_ub(i_trial, :), 1, 'first');
            
            if isempty(lb_1st) && isempty(ub_1st)
                % Crossed neither
                ch1 = cev(i_trial, end) >= 0;
                td1 = nt;
            elseif isempty(lb_1st)
                % Crossed ub only
                ch1 = true;
                td1 = ub_1st;
            elseif isempty(ub_1st)
                % Crossed lb only
                ch1 = false;
                td1 = lb_1st;
            elseif lb_1st < ub_1st
                % Crossed lb first
                ch1 = false;
                td1 = lb_1st;
            else
                % Crossed ub first
                ch1 = true;
                td1 = ub_1st;
            end
            
            ch(i_trial, 1) = ch1;
            td(i_trial, 1) = td1;
        end
        Sim.Ev.ch = ch;
        Sim.Ev.td_fr = td;
        Sim.Ev.rt_fr = td; % Assume zero Tnd
    end
    function cev = get.cev(Sim)
        if isempty(Sim.cev_)
            Sim.cev_ = Sim.get_cev;
        end
        cev = Sim.cev_;
    end
    function cev = get_cev(Sim)
        cev = cumsum(Sim.Ev.ev + Sim.Ev.noise_internal, 2);            
    end
    function set.cev(Sim, v)
        Sim.cev_ = v;
    end
    function set_nan_aft_rt(Sim)
        if Sim.to_set_nan_aft_rt
            ev = Sim.Ev.ev;
            rt = Sim.Ev.rt_fr;
            for tr = 1:Sim.n_trial
                ev(tr,(rt(tr) + 1):end) = nan;
            end
            Sim.Ev.ev = ev;
        end
    end
end
%% Dependent properties
methods
    function v = get.noise_internal(Sim)
        v = Sim.Ev.noise_internal;
    end
    function set.noise_internal(Sim, v)
        Sim.Ev.noise_internal = v;
    end
    
    function v = get.ev(Sim)
        v = Sim.Ev.ev;
    end
    function set.ev(Sim, v)
        Sim.Ev.ev = v;
    end

    function v = get.ch(Sim)
        v = Sim.Ev.ch;
    end
    function set.ch(Sim, v)
        Sim.Ev.ch = v;
    end
end
%% Plot
methods
    function plot_and_save_all(Sim)
        % Do nothing.
    end
end
%% Compatibility
methods
    function varargout = set_root(Sim, varargin)
        [varargout{1:nargout}] = ...
            Sim.set_root@bml_local.oop.PropFileNameTree( ...
                varargin{:});
        [varargout{1:nargout}] = ...
            Sim.set_root@IxnKernel.EvTime.EvTimeSharer( ...
                varargin{:});
    end
end
%% Save
methods
    function fs = get_file_fields0(~)
        fs = {
            'drift', 'dft'
            'sigmasq_ev', 'sqe'
            'sigmasq_internal', 'sqi'
            'bound', 'bnd'
            'n_trial', 'ntr'
            'smooth_ev_args', 'sme'
            'smooth_internal_args', 'smi'
            };
    end
end
end