classdef EvTimeD1 ...
        < IxnKernel.EvTime.EvTimeSharer ...
        & bml_local.oop.PropFileNameTree
    % IxnKernel.EvTime.EvTimeD1 - Allows easy slcing of time and trials    
    
%% Data
properties (Dependent)
    ch % (trial, 1)
    ev % (trial, fr)
    noise_internal % (trial, fr) % Unknown to experimenter
    td_fr % (trial, 1) % frame (= time bin) of the time of first crossing
    rt_fr % (trial, 1) % frame (= time bin) of the time of report
end
properties (Hidden)
    ch_
    ev_
    noise_internal_
    td_fr_
    rt_fr_
end
%% Setting
properties
    tr_incl_ = [];
end
properties (Dependent)
    tr_incl
end
%% Stats
properties (Dependent)
    n_trial
end
%% Init
methods
    function Ev = EvTimeD1(varargin)
        if nargin > 0
            Ev.init(varargin{:});
        end
    end
    function init(Ev, varargin)
        varargin2props(Ev, varargin);
    end
end
%% Trial filter
methods
    function v = get.tr_incl(Ev)
        if isempty(Ev.tr_incl_)
            v = true(size(Ev.ev, 1), 1);
        else
            v = Ev.tr_incl_;
        end
    end
    function set.tr_incl(Ev, v)
        Ev.tr_incl_ = v;
    end
end
%% Trial and evidence filter
methods
    function [ev, ch, S_filt, tr_incl, fr_incl] = get_ev_filtered(Ev, varargin)
        % [ev, ch, S_filt] = get_ev_filtered(Ev, varargin)
        %
        % ev{trial,1}(1,time_bin) : NaN where filtered out.
        % ch(trial, 1) : NaN where filtered out.
        %
        % OPTIONS:
        % ... % Time filter/aligners
        % 'align', 'st'
        % 'interval_ms', [0, 100]
        % 't_dir', []
        % ...
        % ... % Trial filters
        % 'tr_incl', true(Ev.n_trial, 1)
        % 'rt_incl_ms', [1500, 5000]
        % 'filt_tr_kind', 'complete' % 'complete'|'any'
        
        S_filt = Ev.get_S_filt(varargin{:});
        
        % Filter with rt_incl_ms
        % tr_incl(tr,1) = logical
        tr_incl = Ev.get_tr_incl(S_filt);
        
        % Filter with fr
        % fr_incl{tr}(fr)
        fr_incl = Ev.get_fr_incl(S_filt, 'tr_incl', tr_incl);

        % Filter ev and ch
%         ev = row2cell(Ev.ev);
%         ev = ev(tr_incl);
%         fr_incl = fr_incl(tr_incl);
%         ev = cellfun(@(v, fr_incl1) v(fr_incl1), ev, fr_incl, ...
%             'UniformOutput', false);
        ev0 = Ev.ev(tr_incl,:);
        n_tr = nnz(tr_incl);
        
        ev = cell(n_tr, 1);
        for tr = 1:n_tr
            ev{tr} = ev0(tr, fr_incl{tr});
        end
        
        if nargout >= 2
            ch = Ev.ch(tr_incl);
        end
    end
    function tr_incl = get_tr_incl(Ev, varargin)
        S_filt = Ev.get_S_filt(varargin{:});
        
        tr_incl = false(Ev.n_trial, 1);
        tr_incl(S_filt.tr_incl) = true;

        % Filter with rt_incl
        if ~isempty(S_filt.rt_incl_ms) && ~isempty(Ev.rt_fr)
            rt_incl_bin = Ev.Time.convert_sec2fr(S_filt.rt_incl_ms / 1e3);
            
            tr_incl = (Ev.rt_fr >= rt_incl_bin(1)) ...
                    & (Ev.rt_fr < rt_incl_bin(end));
        end
        
        % Combine filters
        tr_incl = tr_incl & S_filt.tr_incl;
    end
    function [fr_incl, st_sec_rel, en_sec_rel] = get_fr_incl(Ev, varargin)
        % [fr_incl, st_sec_rel, en_sec_rel] = get_fr_incl(Ev, varargin)
        % fr_incl{tr,1}(1,fr)
        % st_sec_rel : scalar. st_sec_rel + t0(tr,1) = st_sec(tr,1).
        % en_sec_rel : scalar. en_sec_rel + t0(tr,1) = en_sec(tr,1).
        
        S_filt = Ev.get_S_filt(varargin{:});
        
        % Align
        switch S_filt.align
            case 'st'
                t0_fr = ones(Ev.n_trial, 1);
                if isempty(S_filt.t_dir)
                    S_filt.t_dir = 1;
                end
                
            case 'td'
                t0_fr = Ev.td_fr(:);
                if isempty(S_filt.t_dir)
                    S_filt.t_dir = -1;
                end
                
            case 'rt'
                t0_fr = Ev.rt_fr(:);
                if isempty(S_filt.t_dir)
                    S_filt.t_dir = -1;
                end
        end
        t0_sec = Ev.Time.convert_fr2sec(t0_fr);
        t0_sec = vVec(t0_sec(S_filt.tr_incl));
        
        st_sec_rel = (S_filt.interval_offset_ms/1e3 ...
             + S_filt.interval_spacing_ms/1e3 * (S_filt.interval_ix(1) - 1)) ...
            * S_filt.t_dir;
        en_sec_rel = (S_filt.interval_offset_ms/1e3 ...
             + S_filt.interval_spacing_ms/1e3 * (S_filt.interval_ix(end) - 1) ...
             + S_filt.interval_width_ms/1e3) ...
            * S_filt.t_dir;
        
        st_sec = t0_sec + st_sec_rel;
        en_sec = t0_sec + en_sec_rel;
        
        st_fr = Ev.Time.convert_sec2fr_ix(st_sec);
        en_fr = Ev.Time.convert_sec2fr_ix(en_sec) - S_filt.t_dir;
        
        n_tr = size(st_fr, 1);
        fr_incl = cell(n_tr, 1);
        for tr = 1:n_tr
            fr_incl{tr} = st_fr(tr):S_filt.t_dir:en_fr(tr);
        end
%         fr_incl = arrayfun(@(st, en) st:S_filt.t_dir:en, st_fr, en_fr, ...
%             'UniformOutput', false);
    end
    function S_filt = get_S_filt(Ev, varargin)
        % S_filt = get_S_filt(Ev, varargin)
        % S_filt = get_S_filt(Ev, S_filt, varargin)
        %
        % OPTIONS
        % ... % Time filter/aligners
        % 'align', 'st'
        % 'interval_ms', [0, 100]
        % 't_dir', []
        % ...
        % ... % Trial filters
        % 'tr_incl', true(Ev.n_trial, 1)
        % 'rt_incl_ms', [0, 5000]
        
        if ~isempty(varargin) && isstruct(varargin{1})
            S_filt = varargin{1};
            varargin = varargin(2:end);
        else
            S_filt = struct;
        end
        
        S_filt = varargin2S(varargin, varargin2S(S_filt, {
            ... % Time filter/aligners
            'align', 'rt' % 'st'|'rt'
            't_dir', []
            ...
            ... % Trial filters
            'tr_incl', true(Ev.n_trial, 1)
            'rt_incl_ms', []
            'interval_width_ms', []
            'interval_st_ms', 0
            'interval_spacing_ms', []
            'interval_offset_ms', 0
            ... % interval_ix
            ... % : if nonscalar, first and last elements are used
            ... %   and the rest are ignored.
            'interval_ix', 1
            }));
        if isempty(S_filt.rt_incl_ms)
            S_filt.rt_incl_ms = [0, Ev.max_t * 1e3];
        end
        if isempty(S_filt.interval_width_ms)
            S_filt.interval_width_ms = Ev.max_t * 1e3;
        end
        if isempty(S_filt.interval_spacing_ms)
            S_filt.interval_spacing_ms = S_filt.interval_width_ms;
        end
        if isempty(S_filt.t_dir)
            switch S_filt.align
                case 'st'
                    S_filt.t_dir = 1;
                case 'rt'
                    S_filt.t_dir = -1;
            end
        end
    end
end
%% Intervals
methods
    function S_filt = get_S_filt_intervals(Ev, varargin)
        % Returns default values suitable for iterating over interval_ix.
        % 
        % S_filt = get_S_filt_intervals(~, varargin)
        %
        % OPTIONS:
        % 'align', 'st' % 'st'|'td'rt'
        % 'interval_width_ms', 100
        % 'interval_st_ms', 0
        % 'interval_ix', 1:10
        % 'interval_spacing_ms', 100
        
        C_filt = varargin2C(varargin, {
            'align', 'rt' % 'st'|'td'rt'
            'interval_width_ms', 100
            'interval_st_ms', 0
            'interval_ix', 1:10
            'interval_spacing_ms', 100
            });
        
        S_filt = Ev.get_S_filt(C_filt{:});
    end
    function evs = get_ev_summary_intervals(Ev, varargin)
        % evs = get_ev_summary_intervals(Ev, varargin)
        %
        % evs(tr, itv) = summary of ev in the interval
        %
        % OPTIONS:
        % 'ev', [] % If empty, use Ev.ev
        % 'fun', @(ev, ch) nanmean(v, 2)
        %
        % See also: get_S_filt_intervals
        
        S = varargin2S(varargin, {
            'fun', @(ev, ch) nanmean(ev, 2)
            });
        C = S2C(S);
        
        S_filt = Ev.get_S_filt_intervals(C{:});
        
        n_itv = numel(S_filt.interval_ix);
        evs = cell(n_itv, 1);
        
        for i_itv = 1:n_itv
            C_filt1 = varargin2C({
                'interval_ix', S_filt.interval_ix(i_itv)
                }, S_filt);
            [ev1, ch1] = Ev.get_ev_filtered(C_filt1{:});
            evs{i_itv} = arrayfun(@(ev11, ch11) S.fun(ev11{1}, ch11), ...
                ev1, ch1);
        end
        assert(all(cellfun(@numel, evs) == numel(evs{1})));
        evs = cell2mat2(evs)';
    end
end
%% Stats
methods
    function v = get.n_trial(Ev)
        v = nnz(Ev.tr_incl);
    end
end
%% Test
methods
    function passed = test(Ev)
        passed = true;
        passed = passed && Ev.test_get_tr_incl;
        passed = passed && Ev.test_get_fr_incl;
    end
    function passed = test_get_tr_incl(Ev)
        %%
        n_tr = 5;
        tnd = 2;
        
        Ev.ev = bsxfun(@plus, 10*(1:n_tr)', 1:(n_tr+tnd));
        Ev.td_fr = (1:n_tr)';
        Ev.rt_fr = Ev.td_fr + tnd;
        
        rt_incl_fr = (2:(n_tr - 1)) + tnd;
        rt_incl_ms = Ev.Time.convert_fr2sec(rt_incl_fr) * 1e3;
        
        tr_incl = Ev.get_tr_incl('rt_incl_ms', rt_incl_ms);
        
        disp(tr_incl);
        passed = all(~tr_incl([1, end-1, end])) &  all(tr_incl(2:(end-2)));
    end
    function passed = test_get_fr_incl(Ev)
        %%
        n_tr = 5;
        tnd = 2;
        
        Ev.ev = bsxfun(@plus, 10*(1:n_tr)', 1:(n_tr+tnd));
        Ev.td_fr = (1:n_tr)';
        Ev.rt_fr = Ev.td_fr + tnd;
        
        itv_ix = 1;
        
        fr_incl = Ev.get_fr_incl( ...
            'interval_width_ms', Ev.dt*1e3*1, ...
            'interval_spacing_ms', [], ... Ev.dt*1e3, ...
            'interval_ix', itv_ix, ...
            'align', 'rt');
        
        celldisp(fr_incl);
        
        passed = all(arrayfun(@(c,v) isequal(c{1},v), ...
            fr_incl, (1:n_tr)' + tnd - itv_ix + 1));
    end
    function passed = test_get_ev_filtered(Ev)
        %%
        n_tr = 5;
        tnd = 2;
        
        Ev.ev = bsxfun(@plus, 10*(1:n_tr)', 1:(n_tr+tnd));
        Ev.td_fr = (1:n_tr)';
        Ev.rt_fr = Ev.td_fr + tnd;
        
        Ev.ch = true(n_tr, 1);
        
        rt_incl_fr = (2:(n_tr - 1)) + tnd;
        rt_incl_ms = Ev.Time.convert_fr2sec(rt_incl_fr) * 1e3;
        
        itv_ix = 1;
        
        [ev, ch, S_filt] = Ev.get_ev_filtered( ...
            'interval_width_ms', Ev.dt*1e3*1, ...
            'interval_spacing_ms', [], ... Ev.dt*1e3, ...
            'interval_ix', itv_ix, ...
            'align', 'rt', ...
            'rt_incl_ms', rt_incl_ms);
        
        celldisp(ev);
        disp(ch);
        disp(S_filt);
        
        passed = isequal(ev, {24; 35}) && isequal(ch, true(2,1));
        
%         passed = all(arrayfun(@(c,v) isequal(c{1},v), ...
%             fr_incl, (1:n_tr)' + tnd - itv_ix + 1));
    end
end
%% Get/Set
methods
    function v = get.ch(Ev)
        v = Ev.get_ch;
    end
    function set.ch(Ev, v)
        Ev.set_ch(v);
    end

    function v = get.ev(Ev)
        v = Ev.get_ev;
    end
    function set.ev(Ev, v)
        Ev.set_ev(v);
    end

    function v = get.noise_internal(Ev)
        v = Ev.get_noise_internal;
    end
    function set.noise_internal(Ev, v)
        Ev.set_noise_internal(v);
    end

    function v = get.td_fr(Ev)
        v = Ev.get_td_fr;
    end
    function set.td_fr(Ev, v)
        Ev.set_td_fr(v);
    end

    function v = get.rt_fr(Ev)
        v = Ev.get_rt_fr;
    end
    function set.rt_fr(Ev, v)
        Ev.set_rt_fr(v);
    end


    function v = get_ch(Ev)
        v = Ev.ch_;
    end
    function set_ch(Ev, v)
        Ev.ch_ = v;
    end

    function v = get_ev(Ev)
        v = Ev.ev_;
    end
    function set_ev(Ev, v)
        Ev.ev_ = v;
    end

    function v = get_noise_internal(Ev)
        v = Ev.noise_internal_;
    end
    function set_noise_internal(Ev, v)
        Ev.noise_internal_ = v;
    end

    function v = get_td_fr(Ev)
        v = Ev.td_fr_;
    end
    function set_td_fr(Ev, v)
        Ev.td_fr_ = v;
    end

    function v = get_rt_fr(Ev)
        v = Ev.rt_fr_;
    end
    function set_rt_fr(Ev, v)
        Ev.rt_fr_ = v;
    end
end
end