classdef SimDtbUtilD1 ...
        < IxnKernel.SimDtb.SimDtbD1
    % RevCor.D1.SimDtbUtil - Utilize/ignore part of the evidence stream
    
%% Settings
properties
    % t_util_spec = {kind; arg1; arg2; ...}
    %
    % {'all'}
    % include all. Ignore t_util_args.
    %
    % {'ig_st_en_mod'; [st_ms1, en_ms1, mod_incl1]; ...}
    % ignore {st_ms, en_ms, mod_incl}.
    % mod_incl: used when mod_incl == mod(tr, max(mod_incl) + 1)) 
    %
    % EXAMPLE:
    % {'ig_st_en_mod'; [0, 500, 0]} 
    % : [0, 500] is ignored in all trials
    % 
    % {'ig_st_en_mod'; [0, 500, 0]; [500, 1000, 1]} 
    % : [0, 500] is used in even-numbered trials, and
    %   [500, 1000] is used in odd-numbered trials.
    %
    % {'ig_en'; tnd_ms}
    %   ignore {tnd_ms} from td.
    %
    % {'ig_p_ens'
    %  interval_width_ms
    %  [prct_ignore_interval1, ptct_ignore_inteval2, ...]
    % }
    %   ignore intervalK from the end with prct_ignore_intervalK
    %   by adding Tnd with that probability
    %
    t_util_spec = ...
        {'ig_p_sts'; 100; [30, 30, 70, 70]}; % [100, 0, 100, 0]}; % 
%         {'ig_p_sts'; 100; [100, 100, 0, 0]}; % [100, 0, 100, 0]}; % 
%         {'ig_en'; 200}; 
%         {'ig_st_en_mod', [0, 500, 0]};

    to_debug = false;
    to_set_util_false_aft_rt = true;
end
properties (Dependent)
    t_util_kind % t_util_spec{1}
    t_util_args % t_util_spec(2:end)
end
%% Internal
properties
    % EP: For plotting. IxnKernel.RevCor.D1.Ev2PchD1. 
    EP = []; 
    
    % Util.ev = is_t_util
    % Do not assign handles in the property definition:
    % all instances end up sharing the same instance.
    Util = []; 
end
properties (Dependent)    
    is_t_util % (tr, t) : true if utilized    
end
%% Results
properties
    p_use_sim = []; % (tr, t)
end
%% Init
methods
    function Sim = SimDtbUtilD1(varargin)
        Sim.EP = IxnKernel.RevCor.D1.Ev2PchD1;
        Sim.Util = IxnKernel.EvTime.EvTimeD1; % 
        Sim.add_children_props({'EP', 'Util'}); % Share time & ev
        if nargin > 0
            Sim.init(varargin{:});
        end
    end
end
%% Main
methods
    function simulate(Sim)
        Sim.get_ev;
        Sim.get_noise_internal;

        Sim.get_is_t_util_bef_cev;
        Sim.get_ch_rt;
        
        Sim.get_is_t_util_aft_cev;
        Sim.Util.rt_fr = Sim.Ev.rt_fr;
        Sim.Util.td_fr = Sim.Ev.td_fr;
        Sim.Util.ch = Sim.Ev.ch;
        
        Sim.set_nan_aft_rt;
    end
    function cev = get_cev(Sim)
        cev = cumsum( ...
            (Sim.Ev.ev + Sim.Ev.noise_internal) .* Sim.is_t_util, ...
            2);
    end
    function is_t_util = get_is_t_util_bef_cev(Sim, varargin)
        S = varargin2S(varargin, {
            't_util_kind', Sim.t_util_kind
            't_util_args', Sim.t_util_args
            });
        
        is_t_util = true(Sim.n_trial, Sim.nt);
        p_use = ones(Sim.n_trial, Sim.nt);
        switch S.t_util_kind
            case {'all', 'ig_en'}
                % Do nothing to is_t_util yet
                                
            case 'ig_p_sts'
                [itv_width_ms, prct_ignore] = deal(Sim.t_util_args{:});
                
                n_itv = numel(prct_ignore);
                n_tr = Sim.n_trial;
                n_fr = size(is_t_util, 2);
                p_use_itv = repmat( ...
                    1 - prct_ignore ./ 1e2, ...
                    [n_tr, 1]);
                is_t_util_itv = bsxfun(@lt, ...
                    rand(n_tr, n_itv), ...
                    p_use_itv);
                Ev = Sim.Ev;
                
                for itv = 1:n_itv
                    fr_unused = Ev.get_fr_incl( ...
                        'align', 'st', ...
                        ... % When align=st, all fr_incl{tr} are the same
                        'tr_incl', 1, ... 
                        'interval_width_ms', itv_width_ms, ...
                        'interval_spacing_ms', itv_width_ms, ...
                        'interval_ix', itv);
                    fr_unused = fr_unused{1};
                    out_of_range = ...
                              (fr_unused > n_fr) ...
                            | (fr_unused < 1);
                    fr_unused = fr_unused(~out_of_range);
                    if isempty(fr_unused)
                        continue;
                    end
                    
                    n_fr_unused = numel(fr_unused);
                    
                    is_t_util(:, fr_unused) = ...
                        repmat(is_t_util_itv(:,itv), [1, n_fr_unused]);
                    p_use(:, fr_unused) = ...
                        repmat(p_use_itv(:,itv), [1, n_fr_unused]);
                end
                
            case 'ig_st_en_mod'
                args = cell2mat2(Sim.t_util_args);
                st_ms = args(:,1);
                en_ms = args(:,2);
                
                st_bin = Sim.Time.convert_sec2fr_ix(st_ms / 1e3);
                en_bin = Sim.Time.convert_sec2fr_ix(en_ms / 1e3);
                
                mod_incl = args(:,3);
                max_mod = max(mod_incl) + 1;
                n = size(args, 1);
                
                n_tr = Sim.n_trial;
                t_bin = 1:Sim.nt;
                    
                for ii = 1:n
                    tr_incl = mod(1:n_tr, max_mod) == mod_incl(ii);
                    t_incl = (st_bin <= t_bin) & (t_bin <= en_bin);
                    is_t_util(tr_incl, t_incl) = false;
                end
%                 
%             otherwise
%                 error('Unknown t_util_kind=%s\n', S.t_util_kind);
        end
        
        if nargout == 0
            Sim.is_t_util = is_t_util;
            Sim.p_use_sim = p_use;
        end
    end
    function get_ch_rt(Sim)
        Sim.get_ch_rt@IxnKernel.SimDtb.SimDtbD1;
        
        switch Sim.t_util_kind
            case {'all', 'ig_st_en_mod'}
                % Do nothing
                
            case 'ig_en'
                nt = Sim.nt;
                nt_tnd = Sim.Time.convert_sec2fr_ix( ...
                    Sim.t_util_args{1} / 1e3);
                
                Sim.Ev.rt_fr = min(Sim.Ev.td_fr + nt_tnd, nt);
        end
    end
    function get_is_t_util_aft_cev(Sim)
        is_t_util = Sim.is_t_util;
        
        switch Sim.t_util_kind
            case 'ig_p_ens'
                n_tr = Sim.n_trial;
                
                %% Sample util
                [itv_width_ms, prct_ignore] = deal(Sim.t_util_args{:});
                
                n_itv = numel(prct_ignore);
                is_t_util_itv = ~bsxfun(@lt, ...
                    rand(n_tr, n_itv), ...
                    prct_ignore / 1e2);
                
                %% Generate unused evidence
                % Caveat: not natural in case of nonzero autocorrelation.
                nt_itv = Sim.Time.convert_sec2fr( ...
                    itv_width_ms * n_itv / 1e3);
                ev_unused = Sim.get_ev('nt', nt_itv);
                noise_unused = Sim.get_noise_internal('nt', nt_itv);

                %% Swap used with unused evidence & noise
                Ev = Sim.Ev;
                ev = Ev.ev;
                n_fr = size(ev, 2);
                noise = Ev.noise_internal;
                td_fr = Ev.td_fr;
                
                tr_incl = Sim.Time.convert_fr2sec(td_fr)...
                    >= itv_width_ms * n_itv / 1e3;
                
                for tr = find(tr_incl(:)')
                    ev1 = ev(tr,:);
                    noise1 = noise(tr,:);
                    is_t_util1 = is_t_util(tr,:);
                    
                    is_unused = ~is_t_util_itv(tr,:);
                    td1_fr = td_fr(tr);
                    
                    is_t_util1((td1_fr+1):end) = false;
                    
                    fr_unused_last = 0;
                    
                    for itv = find(is_unused)
                        fr_unused = Ev.get_fr_incl( ...
                            'align', 'rt', ...
                            'tr_incl', tr, ...
                            'interval_width_ms', itv_width_ms, ...
                            'interval_spacing_ms', itv_width_ms, ...
                            'interval_ix', itv - 1);
                        fr_unused = fr_unused{1};
                        out_of_range = ...
                              (fr_unused > n_fr) ...
                            | (fr_unused < 1);
                        fr_unused = fr_unused(~out_of_range);
                        if isempty(fr_unused)
                            continue;
                        end
                        
                        fr_shifted_src = min(fr_unused):td1_fr;
                        fr_shifted_dst = max(fr_unused)+1 ...
                            + fr_shifted_src - min(fr_shifted_src);
                        
                        out_of_range = ...
                              (fr_shifted_dst > n_fr) ...
                            | (fr_shifted_dst < 1) ...
                            | (fr_shifted_src > n_fr) ...
                            | (fr_shifted_src < 1);
                        if any(out_of_range)
                            if all(out_of_range)
                                continue;
                            else
                                fr_shifted_src = fr_shifted_src( ...
                                    ~out_of_range);
                                fr_shifted_dst = fr_shifted_dst( ...
                                    ~out_of_range);
                            end
                        end     
                        
                        n_fr_unused = length(fr_unused);
                        fr_in_unused = fr_unused_last + (1:n_fr_unused);
                        
                        ev1(fr_shifted_dst) = ...
                            ev1(fr_shifted_src);
                        ev1(fr_unused) = ...
                            ev_unused(tr, fr_in_unused);
                        
                        noise1(fr_shifted_dst) = ...
                            noise1(fr_shifted_src);
                        noise1(fr_unused) = ...
                            noise_unused(tr, fr_in_unused);
                        
                        is_t_util1(fr_shifted_dst) = ...
                            is_t_util1(fr_shifted_src);
                        is_t_util1(fr_unused) = false;
                        
                        td1_fr = td1_fr + n_fr_unused;
                        Ev.td_fr(tr) = td1_fr;
                        Ev.rt_fr(tr) = td1_fr; % To update t0 for get_fr_incl.
                        fr_unused_last = fr_in_unused(end);
                        
                        %% Debug
                        if Sim.to_debug
                            plot(is_t_util1);
                            hold on;

                            eprintf any(out_of_range)
                            eprintf td1_fr
                            eprintf n_fr_unused
                            
                            if ~isempty(fr_shifted_src)
                                plot(fr_unused([1, end]), [0 0] + 0.25, 'k-');
                                plot(fr_shifted_src([1, end]), [0 0] + 0.5, 'r-');
                                plot(fr_shifted_dst([1, end]), [0 0] + 0.75, 'b-');
                                
                                eprintf fr_unused([1, end])
                                eprintf fr_shifted_src([1, end])
                                eprintf fr_shifted_dst([1, end])
                            end
                            ylim([-0.05, 1.05]);
                            hold off;
                        end
                    end
                    
                    if td1_fr > n_fr
                        td1_fr = n_fr;
                        ev1 = ev1(1:n_fr);
                        noise1 = noise1(1:n_fr);
                    end
                    
                    ev(tr, :) = ev1;
                    noise(tr, :) = noise1;
                    is_t_util(tr, :) = is_t_util1;
                    td_fr(tr) = td1_fr;
                end                
                for tr = 1:Sim.Ev.n_trial
                    is_t_util(tr, (td_fr(tr)+1):end) = false;
                end
                
                %% DEBUG
                if Sim.to_debug
                    imagesc(is_t_util);
                end
                Sim.Ev.ev = ev;
                Sim.Ev.noise_internal = noise;
                Sim.Ev.td_fr = td_fr;
                Sim.Ev.rt_fr = td_fr;
                
            otherwise
                if Sim.to_set_util_false_aft_rt
                    td = Sim.Ev.td_fr;
                    for tr = 1:Sim.Ev.n_trial
%                         is_t_util(tr, (td(tr)+1):end) = false;
                    end                
                end
        end
        
        Sim.is_t_util = is_t_util;
    end
    function v = get.t_util_kind(Sim)
        v = Sim.t_util_spec{1};
    end
    function v = get.t_util_args(Sim)
        v = Sim.t_util_spec(2:end);
    end
end
%% interval summary
methods
    function ev = get_ev_summary_intervals(Sim, varargin)
        ev(:,:) = Sim.Ev.get_ev_summary_intervals( ...
            varargin{:});
    end
end
%% is_t_util
methods
    function utils = get_util_intervals(Sim, varargin)
        % evs = get_util_intervals(Sim, varargin)
        %
        % evs(tr, itv) = summary of ev in the interval
        %
        % OPTIONS:
        % 'fun', @(ev, ch) nanmean(v, 2)
        %
        % See also: get_S_filt_intervals
        
        utils = Sim.Util.get_ev_summary_intervals(varargin{:});
    end
    function set.is_t_util(Sim, v)
        Sim.Util.ev = double(v);
    end
    function v = get.is_t_util(Sim)
        v = Sim.Util.ev;
    end
end
%% Plot
methods
    function plot_and_save_all(Sim)
        %%
        Sim.EP.Ev = Sim.Ev;
        Sim.EP.main;
    end
    function imagesc_util(Sim)
        Sim.imagesc_t_tr(Sim.is_t_util);
    end
    function imagesc_p_use(Sim)
        Sim.imagesc_t_tr(Sim.p_use_sim);
    end
    function imagesc_t_tr(Sim, v)
        imagesc(Sim.t * 1e3, 1:Sim.n_trial, v);
        colormap(gray);
        set(gca, 'CLim', [0, 1], 'TickDir', 'out');
        xlabel('Time (ms)');
        ylabel('Trial');
    end
end
%% Save
methods
    function fs = get_file_fields0(Sim)
        fs = [
            Sim.get_file_fields0@IxnKernel.SimDtb.SimDtbD1
            {
            't_util_spec', 'tutl'
            }];
    end
end
end