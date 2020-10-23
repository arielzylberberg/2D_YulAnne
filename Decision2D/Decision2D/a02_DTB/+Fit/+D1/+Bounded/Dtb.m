classdef Dtb < Fit.Common.CommonWorkspace
    % Fit.D1.Bounded.Dtb
    %
    % Given parameters k, b, and conditions, produce Td_pdf
    %
    % 2015 YK wrote the initial version.
%% Settings
properties (Dependent)
    fix_sigmaSq_st % Whether to model starting point variability
end
properties
    kbratio_kind = 'bkl';
    calc_unabs = false;
end
%% Internal
properties (SetAccess = protected)
    Drift % Gets k, bias, etc.
    Bound % Gets b, bias, etc.
    SigmaSq
end
properties (Transient)
    Td_pdf % t x nCond x ch
    unabs  % y x t x nCond
    S_dtb  % Command to dtb.pred.calc_dtb
end
properties (Dependent)
    sigmaSq_st
end
properties (Constant)
    MIN_SIGMASQ_ST = 1e-6;
end
%% Methods
methods
    function W = Dtb(varargin)
    %     W.empty_on_save({'Td_pdf', 'unabs'});
        W.add_deep_copy({'Drift', 'Bound', 'EvAxis', 'Data'});

        % Set default drift and bound
        W.set_Data;
        W.set_Drift;
        W.set_Bound;
        W.set_SigmaSq;

        W.fix_sigmaSq_st = true;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.Common.CommonWorkspace(varargin{:});
        
        switch W.kbratio_kind
            case {'', 'n'}
                W.remove_params({
                    'kb'
                    'bkratio'
                    'log10_kb'
                    'log10_bkratio'
                    });
                W.Drift.remove_params_all;
                W.Drift.init_params0;
                W.Bound.remove_params_all;
                W.Bound.init_params0;
                
            case {'bk', 'bkl'}
                W.Drift.fix_to_th0_('k');
                W.Bound.fix_to_th0_('b');
                k0 = W.Drift.th0.k;
                b0 = W.Bound.th0.b;
                
                kb0 = k0 * b0;
                kb_lb = 1; % 0.1;
                kb_ub = 70;
                
                bkratio0 = b0 / k0;
                bkratio_lb = 0.01;
                bkratio_ub = 1;
                
                switch W.kbratio_kind
                    case 'bk'
                        W.add_params({
                            {'kb', kb0, kb_lb, kb_ub}
                            {'bkratio', bkratio0, bkratio_lb, bkratio_ub}
                            });
                    case 'bkl'
                        W.add_params({
                            {'log10_kb', ...
                                log10(kb0), log10(kb_lb), log10(kb_ub)}
                            {'log10_bkratio', ...
                                log10(bkratio0), ...
                                log10(bkratio_lb), log10(bkratio_ub)}
                            });                
                        W.add_constraints({
                            {'A', {'log10_kb', 'log10_bkratio'}, ...
                                {[0.5, 0.5], log10(3)}} % log10(b) <= 
                            {'A', {'log10_kb', 'log10_bkratio'}, ...
                                {-[0.5, 0.5], -log10(0.1)}} % log10(b) >= 
                            {'A', {'log10_kb', 'log10_bkratio'}, ...
                                {[0.5, -0.5], log10(70)}} % log10(k) <= 
                            {'A', {'log10_kb', 'log10_bkratio'}, ...
                                {-[0.5, -0.5], -log10(2)}} % log10(k) >= 
                            });
                end
            otherwise
                error('Illegal kbratio_kind=%s\n', W.kbratio_kind);
        end
    end
    function pred(W)
        % See also: get_Td_pdf
        W.Data.set_Td_pred_pdf(W.get_Td_pdf);
    end
    function use_kbratio(W)
        switch W.kbratio_kind
            case {'', 'n'}
                % Do nothing
                
            case {'bk', 'bkl'}
                switch W.kbratio_kind
                    case 'bk'
                        kb = W.th.kb;
                        bkratio = W.th.bkratio;
                        
                    case 'bkl'
                        kb = 10. ^ W.th.log10_kb;
                        bkratio = 10 .^ W.th.log10_bkratio;
                end
                
                k = sqrt(kb / bkratio);
                b = sqrt(kb * bkratio);
                
                W.Drift.th.k = k;
                W.Drift.fix_to_th_('k');
                W.Bound.th.b = b;
                W.Bound.fix_to_th_('b');
            otherwise
                error('Illegal kbratio_kind=%s\n', W.kbratio_kind);
        end
    end
    function set.fix_sigmaSq_st(W, v)
        if v
            v0 = log10(W.MIN_SIGMASQ_ST);
            W.add_params({
                {'log10_sigmaSq_st', v0, v0, v0}
                });
        else
            W.add_params({
                {'log10_sigmaSq_st', -4, log10(W.MIN_SIGMASQ_ST), 0}
                });
        end
    end
    function v = get.fix_sigmaSq_st(W)
        v = W.th_fix.log10_sigmaSq_st;
    end
    function v = get.sigmaSq_st(W)
        v = 10 .^ W.th.log10_sigmaSq_st;
    end
    function set.sigmaSq_st(W, v)
        W.th.log10_sigmaSq_st = log10(v);
    end
    function fs = get_file_fields0(W)
        fs = {
            'fix_sigmaSq_st', 'fsqs'
            };
    end
end
%% Internal - Td_pdf and unabs
methods
    function [Td_pdf, unabs, S] = get_Td_pdf(W, varargin)
        % [Td_pdf, unabs, S] = get_Td_pdf(W)

        W.use_kbratio;
        
        drift_cond_t = W.get_drift_cond_t;    
        if isempty(drift_cond_t)
            Td_pdf = W.get_Td_pdf_placeholder;
            unabs  = W.get_unabs_placeholder;
            return;
        end

        bound_t_ch = W.get_bound_t_ch;
        y = W.EvAxis.determine_y;
        t = W.get_t;

        S = varargin2S(varargin, {
            'calc_unabs', W.calc_unabs
            'sigmaSq', W.get_sigmaSq
            'sigmaSq_st', W.sigmaSq_st
            });
        C = S2C(S);
        
        [Td_pdf, unabs, S] = W.calc_dtb(drift_cond_t, t, bound_t_ch, y, ...
            C{:});

        % Td_pdf
        W.check_Td_pdf(Td_pdf);

        % unabs
        if W.calc_unabs
            W.check_unabs(unabs);
            W.unabs = unabs;
        end    
        
        % S
        W.S_dtb = S;
    end
    function check_Td_pdf(W, Td_pdf)
        if nargin < 2, Td_pdf = W.Td_pdf; end
        assert(isequal(size(Td_pdf), [W.Time.get_nt, W.get_n_drift, 2]));
    end
    function Td_pdf = get_Td_pdf_placeholder(W)
        Td_pdf = nan(W.get_n_drift, W.Time.get_nt, 2);
    end
end
methods (Static)
    function S = get_dim_names_Td_pdf
        S = varargin2S({
            't', 1
            'drift', 2
            'ch', 3
            });
    end
end
methods
    function check_unabs(W, unabs)
        if nargin < 2, unabs = W.unabs; end
        if W.calc_unabs
            assert(isequal(size(unabs), [W.Time.get_nt, W.get_ny, W.get_n_drift]));
        else
            assert(isempty(W.unabs));
        end
    end
    function unabs = get_unabs_placeholder(W)
        unabs = nan(W.get_ny, W.Time.get_nt, W.get_n_drift);
    end
end
methods (Static)
    function S = get_dim_names_unabs
        S = varargin2S({
            'y', 1
            't', 2
            'drift', 3
            });
    end
end
methods
    %% Data interface
    function v = get_conds_rel(W)
        v = W.Data.get_conds_rel;
    end
    %% Drift interface
    function drift_cond_t = get_drift_cond_t(W)
        drift_cond_t = W.Drift.get_drift_cond_t;

        % n_row doesn't equal nCondsRel for En
    %     nCondsRel = W.Data.get_nConds_rel;
    %     assert(isequal(size(drift_cond_t), [nCondsRel, W.nt]));
    end
    function drift_vec = get_drift_vec(W)
        drift_vec = W.Drift.get_drift_vec;
    end
    function v = get_n_drift(W)
        v = size(W.get_drift_cond_t, 1);
    end
    %% SigmaSq interface
    function sigmaSq = get_sigmaSq(W)
        sigmaSq = W.get_sigmaSq_vec;
    end
    function sigmaSq_vec = get_sigmaSq_vec(W)
        if ~isempty(W.SigmaSq)
            sigmaSq_vec = W.SigmaSq.get_sigmaSq_vec;
        else
            sigmaSq_vec = 1;
        end
        if isscalar(sigmaSq_vec)
            sigmaSq_vec = sigmaSq_vec + zeros(W.get_n_drift, 1);
        end
    end
    %% EvAxis interface
    function ny = get_ny(W)
        [~,~,~,ny] = W.EvAxis.determine_y;
    end
    %% Bound interface
    function bound_t_ch = get_bound_t_ch(W)
        % bound_t_ch: nt x [lb, ub]
        %
        % bounds are common to all conditions.
        bound_t_ch = W.Bound.get_bound_t_ch;
        assert(isequal(size(bound_t_ch), [W.nt, 2])); % nt x [lb, ub]
    end
    %% Internal - object properties
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D1.Common.DataChRtPdf; end
        obj_or_name = W.enforce_class('Fit.D1.Common.DataChRtPdf', obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end
    function set_Drift(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Drift = W.enforce_class('Fit.D1.Bounded.Drift', obj_or_name);
        W.set_sub_from_props({'Drift'});
    end
    function set_Bound(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Bound = W.enforce_class('Fit.D1.Bounded.Bound', obj_or_name);
        W.set_sub_from_props({'Bound'});
    end
    function set_SigmaSq(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.SigmaSq = W.enforce_class('Fit.D1.Bounded.SigmaSq', obj_or_name);
        W.set_sub_from_props({'SigmaSq'});
    end
end
%% Demo
methods (Static)
    function Dtb = demo
        Dtb = Fit.D1.Bounded.Dtb;
%         Data = Fit.D1.Common.DataChRtPdf.demo;
%         Data.load_data;
%         Dtb.set_Data(Data);
        Dtb.Data.set_path;
        Dtb.Data.load_data;
    end
    function Dtb = demo_calc_dtb(Dtb, varargin)
        % UNDER CONSTRUCTION
        S = varargin2S(varargin, {
            'is_sigmaSq_dep_on_drift', false
            'sigmaSq_fun', @(cond, slope, offset) abs(slope) .* cond + offset
            'sigmaSq_slope', 1
            'sigmaSq_offset', 1
            'sigmaSq', 1
            });
        if S.is_sigmaSq_dep_on_drift
            S.sigmaSq = S.sigmaSq_fun(W.get_drift_vec, ...
                S.sigmaSq_slope, S.sigmaSq_offset);
        end
        
    end
end
%% Adaptor
methods (Static)
    function varargout = calc_dtb(drift_cond_t, t, bound_t_ch, y, varargin)
        C = varargin2C(varargin, {
            'is_constant_bound', false % enforce spectral for consistency
            });
        
        
        [varargout{1:nargout}] = dtb.pred.calc_dtb( ...
            drift_cond_t, t, bound_t_ch, y, C{:});
    end
end
end