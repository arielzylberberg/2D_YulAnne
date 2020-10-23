classdef SimDtbUtilD2 ...
        < IxnKernel.EvTime.CommonWorkspaceD2
    % IxnKernel.RevCor.D2.SimDtbUtil D2
    % Use part of the evidence streams jointly with specified probability.
    
%% Settings
properties
    % t_util_spec = {kind; arg1; arg2; ...}
    %
    % {'all'}
    % include all. Ignore t_util_args.
    %
    % {'jt_st_sym'; interval_width_ms; [prct_jt_itv1, prct_jt_itv2, ...]}
    %  Use jointly with the given probability.
    %  There is no probability that none is used,
    %  and there are equal probabilities that either stream is used alone.
    %
    % {'itv_bmc'; interval_number1; [both1, motion_only1, color_only1]; ...}
    %  Others are all both = 100.
    %  Interval width is assumed to be 100ms.
    t_util_spec = {
        'itv_bmc'
        1
        [70, 30, 0]
        2 % [2, 3]
        [0, 70, 30]
        };
%         {'jt_st_sym'; 100; [50, 50, 0, 0]};

    to_debug = false;
end
properties (Dependent)
    t_util_kind % t_util_spec{1}
    t_util_args % t_util_spec(2:end)
    
    n_trial
    % nt % Defined in IxnKernel.EvTime.TimeAxis.TimeInheritable
end
%% Internal
properties
    SimDtbs
end
properties (Dependent)
    noise_internal % (tr, t, dim)
    is_t_util % (tr, t, dim) : true if utilized
    cev % (tr, t, dim)
    
    ev % (tr, t, dim)
    ch % (tr, dim)
end
%% Init
methods
    function Sim = SimDtbUtilD2(varargin)
        Sim.SimDtbs = {IxnKernel.RevCor.D1.SimDtbUtilD1, IxnKernel.RevCor.D1.SimDtbUtilD1};
        Sim.add_children_props({'SimDtbs'}); % Share time & ev
        Sim.props_to_share_Ev = {'SimDtbs'};
        
        if nargin > 0
            Sim.init(varargin{:});
        end
    end
    function init(Sim, varargin)
        varargin2props(Sim, varargin, true);
        for dim = 1:Sim.n_dim
            Sim.SimDtbs{dim}.init(varargin{:});
        end
    end
end
%% Main
methods
    function simulate(Sim)
        Sim.calc_ev;
        Sim.calc_noise_internal;

        Sim.get_is_t_util_bef_cev;
        Sim.get_ch_rt;
        
        Sim.get_is_t_util_aft_cev;        
        Sim.Ev2Util;
        
        Sim.set_nan_aft_rt;
    end
    function calc_ev(Sim)
        for ii = 1:Sim.n_dim
            Sim.SimDtbs{ii}.get_ev;
        end
    end
    function calc_noise_internal(Sim)
        for ii = 1:Sim.n_dim
            Sim.SimDtbs{ii}.get_noise_internal;
        end
    end
    function is_t_util = get_is_t_util_bef_cev(Sim, varargin)
        S = varargin2S(varargin, {
            't_util_kind', Sim.t_util_kind
            't_util_args', Sim.t_util_args
            });
        
        is_t_util = true(Sim.n_trial, Sim.nt, Sim.n_dim);
        switch S.t_util_kind
            case 'all'
                % Do nothing to is_t_util yet
                
            case 'jt_st_sym'
                %%
                [itv_width_ms, prct_jt] = deal(S.t_util_args{:});
                
                n_itv = numel(prct_jt);
                n_tr = Sim.n_trial;
                n_fr = size(is_t_util, 2);

                p_jt = prct_jt(:) / 100;
                p_use = [p_jt, (1 - p_jt) / 2, (1 - p_jt) / 2];
                p_use = [p_use, 1 - sum(p_use, 2)];
                
                for itv = 1:n_itv
                    p1 = p_use(itv, :);
                    is_itv_util = mnrnd(1, p1, n_tr); % (:, [jt, 1_only, 2_only, none])
                    
                    for dim = Sim.n_dim:-1:1
                        is_itv_util_dim = ...
                            sum(is_itv_util(:,[1, dim + 1]), 2);
                        
                        Ev = Sim.SimDtbs{dim}.Ev;
                        
                        fr_incl = Ev.get_fr_incl( ...
                            'align', 'st', ...
                            ... % When align=st, all fr_incl{tr} are the same
                            'tr_incl', 1, ... 
                            'interval_width_ms', itv_width_ms, ...
                            'interval_spacing_ms', itv_width_ms, ...
                            'interval_ix', itv);
                        fr_incl = fr_incl{1};
                        out_of_range = ...
                                  (fr_incl > n_fr) ...
                                | (fr_incl < 1);
                        fr_incl = fr_incl(~out_of_range);
                        if isempty(fr_incl)
                            continue;
                        end

                        n_fr_incl = numel(fr_incl);

                        is_t_util(:, fr_incl, dim) = ...
                            repmat(is_itv_util_dim, [1, n_fr_incl]);
                    end
                end
                
            case 'itv_bmc'
                % {'itv_bmc'; interval_number1; [both1, motion_only1, color_only1]; ...}
                %  Others are all both = 100.
                args = S.t_util_args;
                n_itv = numel(args) / 2;
                
                ix_itv = args(1:2:end);
                itv_max = max(cell2vec(ix_itv));
                
                p_use = cell2mat2(args(2:2:end)) / 100;
                p_use(:,4) = 1 - sum(p_use, 2);
                
                n_tr = Sim.n_trial;
                n_fr = size(is_t_util, 2);
                
                itv_width_ms = 100;
                
                for i_itv = 1:n_itv
                    itvs = ix_itv{i_itv};
                    for itv = itvs(:)'
                        p1 = p_use(i_itv, :);
                        is_itv_util = mnrnd(1, p1, n_tr); % (:, [jt, 1_only, 2_only, none])

                        for dim = Sim.n_dim:-1:1
                            is_itv_util_dim = ...
                                sum(is_itv_util(:,[1, dim + 1]), 2);

                            Ev = Sim.SimDtbs{dim}.Ev;

                            fr_incl = Ev.get_fr_incl( ...
                                'align', 'st', ...
                                ... % When align=st, all fr_incl{tr} are the same
                                'tr_incl', 1, ... 
                                'interval_width_ms', itv_width_ms, ...
                                'interval_spacing_ms', itv_width_ms, ...
                                'interval_ix', itv);
                            fr_incl = fr_incl{1};
                            out_of_range = ...
                                      (fr_incl > n_fr) ...
                                    | (fr_incl < 1);
                            fr_incl = fr_incl(~out_of_range);
                            if isempty(fr_incl)
                                continue;
                            end

                            n_fr_incl = numel(fr_incl);

                            is_t_util(:, fr_incl, dim) = ...
                                repmat(is_itv_util_dim, [1, n_fr_incl]);
                        end
                    end
                end
        end
        
        if nargout == 0
            Sim.is_t_util = is_t_util;
            
            % Check:
            %   Sim.SimDtbs{1}.Util must be different from
            %   Sim.SimDtbs{2}.Util.
            is_t_util = Sim.is_t_util;
            for ii = 1:2
                subplot(1,2,ii); 
                imagesc(is_t_util(:,:,ii)); 
                title(sprintf('Dim %d', ii));
                
                if ii == 1
                    xlabel('Frame');
                    ylabel('Trial');
                end
            end
        end
    end    
    function cev = get_cev(Sim)
        % cev(tr, fr, dim)
        cev = cumsum( ...
            (Sim.ev + Sim.noise_internal) .* Sim.is_t_util, ...
            2);
    end
    function get_ch_rt(Sim)
        for ii = 1:Sim.n_dim
            Sim.SimDtbs{ii}.get_ch_rt;
        end
        
        %% Set RT to the maximum of the two
        rt_fr = max(Sim.SimDtbs{1}.Ev.rt_fr, Sim.SimDtbs{2}.Ev.rt_fr);
        
        for dim = 1:numel(Sim.SimDtbs)
            Sim.SimDtbs{dim}.Ev.rt_fr = rt_fr;
        end
    end
    
    function get_is_t_util_aft_cev(~)
        % Nothing to do for current values of t_util_kind
    end
    function v = get.t_util_kind(Sim)
        v = Sim.t_util_spec{1};
    end
    function v = get.t_util_args(Sim)
        v = Sim.t_util_spec(2:end);
    end
    
    function v = get.noise_internal(Sim)
        for dim = Sim.n_dim:-1:1
            v(:,:,dim) = Sim.SimDtbs{dim}.Ev.noise_internal;
        end
    end
    function set.noise_internal(Sim, v)
        for dim = 1:Sim.n_dim
            Sim.SimDtbs{dim}.noise_internal = v(:,:,dim);
        end
    end
    
    function v = get.ev(Sim)
        for dim = Sim.n_dim:-1:1
            v(:,:,dim) = Sim.SimDtbs{dim}.Ev.ev;
        end
    end
    function set.ev(Sim, v)
        for dim = 1:Sim.n_dim
            Sim.SimDtbs{dim}.ev = v(:,:,dim);
        end
    end

    function v = get.ch(Sim)
        for dim = Sim.n_dim:-1:1
            v(:,dim) = Sim.SimDtbs{dim}.Ev.ch;
        end
    end
    function set.ch(Sim, v)
        for dim = Sim.n_dim:-1:1
            Sim.SimDtbs{dim}.Ev.ch = v(:,dim);
        end
    end
    
    function Ev2Util(Sim0)
        for dim = 1:Sim0.n_dim
            Sim = Sim0.SimDtbs{dim};
            
            Sim.Util.rt_fr = Sim.Ev.rt_fr;
            Sim.Util.td_fr = Sim.Ev.td_fr;
            Sim.Util.ch = Sim.Ev.ch;
        end
    end
    function set_nan_aft_rt(Sim)
        for ii = 1:Sim.n_dim
            Sim.SimDtbs{ii}.set_nan_aft_rt;
        end
    end
    
    function v = get.n_trial(Sim)
        v = Sim.SimDtbs{1}.n_trial;
    end
    function v = get_nt(Sim)
        v = Sim.SimDtbs{1}.nt;
    end
end
%% interval summary
methods
    function ev = get_ev_summary_intervals(Sim, varargin)
        for dim = Sim.n_dim:-1:1
            ev(:,:,dim) = Sim.Ev.Evs{dim}.get_ev_summary_intervals( ...
                varargin{:});
%             ev(:,:,dim) = Sim.SimDtbs{dim}.Ev.get_ev_summary_intervals( ...
%                 varargin{:});
        end
    end
end
%% is_t_util
methods
    function utils = get_util_intervals(Sim, varargin)
        % utils = get_util_intervals(Sim, varargin)
        %
        % utils(tr, itv, dim) = summary of ev in the interval
        %
        % OPTIONS:
        % 'fun', @(ev, ch) nanmean(v, 2)
        %
        % See also: get_S_filt_intervals
        
        for dim = Sim.n_dim:-1:1
            utils(:,:,dim) = Sim.SimDtbs{dim}.get_util_intervals( ...
                varargin{:});
        end
    end
    function set.is_t_util(Sim, v)
        for dim = 1:Sim.n_dim
            Sim.SimDtbs{dim}.is_t_util = double(v(:,:,dim));
        end
    end
    function v = get.is_t_util(Sim)
        for dim = Sim.n_dim:-1:1
            v(:,:,dim) = Sim.SimDtbs{dim}.is_t_util;
        end
    end
end
%% Plot
methods
    function plot_and_save_all(Sim)
        %%
        Sim.EP.Ev = Sim.Ev;
        Sim.EP.main;
    end
end
%% Save
methods
    function fs = get_file_fields0(Sim)
        fs = [
            Sim.get_file_fields0@IxnKernel.EvTime.CommonWorkspace
            {
            't_util_spec', 'tutl'
            }];
    end
end
end