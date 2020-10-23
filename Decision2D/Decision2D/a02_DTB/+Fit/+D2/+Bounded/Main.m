classdef Main < Fit.D2.Common.Main
    % Fit.D2.Bounded.Main
    %
    % 2015 YK wrote the initial version.
%% Settings
properties (Dependent)
    td_kind % Linked to Dtb
end
%% Internal
properties (SetAccess = protected)
    Dtb    
    Tnd
%     Miss
end
properties (Dependent)
    Drifts
    Drift1
    Drift2
    Bounds
    Bound1
    Bound2
    KBRatios
    KBRatio1
    KBRatio2
    SigmaSq1
    SigmaSq2
    SigmaSqs
    Drift1_kind
    Drift2_kind
    Bound1_kind
    Bound2_kind
    SigmaSq1_kind
    SigmaSq2_kind
    Tnd_kind
end
properties
    force_repeated_pred = false
end
%% Main
methods
    function W = Main(varargin)
        W.add_deep_copy({'Dtb', 'Tnd', 'Miss'});

        W.set_Data;
        W.set_Dtb;
        W.set_Tnd;
        W.set_Miss;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function pred(W)
        persistent th_Dtb_bef
        
        if ~isequal(W.Dtb.th_vec, th_Dtb_bef) ...
                || isempty(W.Data.Td_pred_pdf) || W.force_repeated_pred
            th_Dtb_bef = W.Dtb.th_vec;
            W.Dtb.pred;
        else
%             disp('Skipped DTB!'); % DEBUG
        end
        W.Tnd.pred;
        W.Miss.pred;
        
%         if ~W.to_include_last_frame
%             % Normalize pred after removing the last frame
%             W.Data.RT_pred_pdf = ...
%                 W.set_last_frame_0_and_normalize(W.Data.RT_pred_pdf);
%         end
    end
    function v = get_file_fields0(W)
        v = union_general(W.get_file_fields0@Fit.D2.Common.Main, ...
            {
            'td_kind', 'td'
            }, 'stable', 'rows');            
    end
end
%% User interface
methods (Static)
    function W = demo_get_W(set_path_args)
        if nargin < 1
            set_path_args = {}; 
        else
            % Enforce cell
            set_path_args = varargin2C(set_path_args);
        end
        
        W = Fit.D2.Bounded.Main;
        W.Data.set_path(set_path_args, 'A', 1);
        W.Data.load_data;
    end
end
methods (Static)
    function Fl = demo_fit_Fl(Fl, varargin)
        S = varargin2S(varargin, {
            'to_plot', true
            'UseParallel', 'always'
            });
        
        %% Add plotfun
%         Fl.remove_plotfun_all;
%         Fit.D2.Common.Plot.PlotFuns.add_plotfun(Fl);
        Fl.plot_opt.to_plot = S.to_plot;
        
        %% Test run
        Fl.W.pred;
        Fl.runPlotFcns;
        
        %% Test run
        tic;
        c = Fl.get_cost(Fl.W.th_vec);
        disp(c);
        toc;
        
        %% Fit
        Fl.fit('opts', {'UseParallel', S.UseParallel});
    end
    function [Fl, W] = fit_aft_create_inh(varargin)
        C = varargin2C(varargin, {
            'Td', ''
            });
        [W, Fl] = Fit.D2.Bounded.Main.demo_get_W_inh(C{:});
        Fl = Fit.D2.Bounded.Main.demo_fit_Fl(Fl);
    end
    function [Fl, W] = fit_aft_create(varargin)
        % [Fl, W] = fit_aft_create(varargin)
        %
        % Also arguments for Data.DataLocator: 'subj', 'parad' 
        S = varargin2S(varargin, {
            'Td', 'Ser'
            });
        C = S2C(S);
        
        W = Fit.D2.Bounded.Main.demo_get_W(S); % C{:});
        if ~isempty(S.Td)
            W.Dtb.set_Td(S.Td);
        end
        
        %%
        Fl = W.get_Fl;
        
        Fl = Fit.D2.Bounded.Main.demo_fit_Fl(Fl);
    end
    function varargout = demo(varargin)
        [varargout{1:nargout}] = ...
            Fit.D2.Bounded.Main.fit_unit(varargin{:});
    end
end
%% Init / BatchFit interface
methods
%     function Fl = get_Fl(W, varargin)
% %         res = struct;
% %         W.init(varargin{:});
% 
%         %% Get Fl
%         Fl = W.get_Fl@Fit.D2.Common.Main;
%     end
    function [Fl, S, res] = fit_unit(W, varargin)
        % [Fl, S, res] = fit_unit(W, varargin)
        
        %% Get Fl
        [Fl, S, res] = W.get_Fl(varargin{:});
        
        %% Fit
        Fl.fit(S.fit_args{:});
        
        %%
        disp('res.th');
        disp(Fl.res.th);
        disp('res.se');
        disp(Fl.res.se);
    end
end
%% Bias
methods
    function b = get_cond_bias(W)
        b = {W.Drifts{1}.get_cond_bias, W.Drifts{2}.get_cond_bias};
    end
end
%% Object properties
methods
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = 'IndepDim'; end
        % FIXIT: use a factory rather than the default class.
        
        if ~isempty(W.Dtb)
            C = {'td_kind', W.Dtb.td_kind};
        else
            C = {};
        end
        W.Dtb = W.enforce_class('Fit.D2.Bounded.Dtb', obj_or_name);
        varargin2props(W.Dtb, C);
        W.set_sub_from_props({'Dtb'});
    end
    function set_Tnd(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Tnd = W.enforce_class('Fit.D2.Common.Tnd', obj_or_name);
        W.set_sub_from_props({'Tnd'});
    end
    
    % Drift
    function set_Drift(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        for dim = 1:2
            W.Dtb.Dtbs{dim}.set_Drift(kind);
        end
    end
    function set_Drift1(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        W.Dtb.Dtb1.set_Drift(kind);
    end
    function set_Drift2(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        W.Dtb.Dtb2.set_Drift(kind);
    end
    
    % Bound
    function set_Bound(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        for dim = 1:2
            W.Dtb.Dtbs{dim}.set_Bound(kind);
        end
    end
    function set_Bound1(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        W.Dtb.Dtb1.set_Bound(kind);
    end
    function set_Bound2(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        W.Dtb.Dtb2.set_Bound(kind);
    end
    
    % SigmaSq
    function set_SigmaSq(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        for dim = 1:2
            W.Dtb.Dtbs{dim}.set_SigmaSq(kind);
        end
    end
    function set_SigmaSq1(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        W.Dtb.Dtb1.set_SigmaSq(kind);
    end
    function set_SigmaSq2(W, kind)
        if ~exist('kind', 'var'), kind = ''; end
        W.Dtb.Dtb2.set_SigmaSq(kind);
    end
end
%% Object dependent properties
methods
    function v = get.Drifts(W)
        v = {W.Drift1, W.Drift2};
    end
    function v = get.Drift1(W)
        v = W.get_Drift1;
    end
    function set.Drift1(W, v)
        W.set_Drift1(v);
    end
    function v = get.Drift2(W)
        v = W.get_Drift2;
    end
    function set.Drift2(W, v)
        W.set_Drift2(v);
    end
    
    function v = get.Bounds(W)
        v = {W.Bound1, W.Bound2};
    end
    function v = get.Bound1(W)
        v = W.get_Bound1;
    end
    function set.Bound1(W, v)
        W.set_Bound1(v);
    end
    function v = get.Bound2(W)
        v = W.get_Bound2;
    end
    function set.Bound2(W, v)
        W.set_Bound2(v);
    end
    
    function v = get.KBRatios(W)
        v = W.Dtb.KBRatios;
    end
    function set.KBRatios(W, v)
        W.Dtb.KBRatios = v;
    end

    function v = get.SigmaSq1(W)
        v = W.get_SigmaSq1;
    end
    function set.SigmaSq1(W, v)
        W.set_SigmaSq1(v);
    end
    function v = get.SigmaSq2(W)
        v = W.get_SigmaSq2;
    end
    function set.SigmaSq2(W, v)
        W.set_SigmaSq2(v);
    end
    function v = get.SigmaSqs(W)
        v = W.Dtb.SigmaSqs;
    end
    function set.SigmaSqs(W, v)
        W.Dtb.SigmaSqs = v;
    end
    
    function v = get_Drift1(W)
        v = W.Dtb.Dtb1.Drift;
    end
    function v = get_Drift2(W)
        v = W.Dtb.Dtb2.Drift;
    end
    function v = get_SigmaSq1(W)
        v = W.Dtb.Dtb1.SigmaSq;
    end
    function v = get_SigmaSq2(W)
        v = W.Dtb.Dtb2.SigmaSq;
    end
    function v = get_Bound1(W)
        v = W.Dtb.Dtb1.Bound;
    end
    function v = get_Bound2(W)
        v = W.Dtb.Dtb2.Bound;
    end
end
%% Object kinds
methods
    function v = get.Drift1_kind(W)
        v = W.get_Drift1_kind;
    end
    function set.Drift1_kind(W, v)
        W.set_Drift1(v);
    end

    function v = get.Drift2_kind(W)
        v = W.get_Drift2_kind;
    end
    function set.Drift2_kind(W, v)
        W.set_Drift2(v);
    end
    
    function v = get.Bound1_kind(W)
        v = W.get_Bound1_kind;
    end
    function set.Bound1_kind(W, v)
        W.set_Bound1(v);
    end

    function v = get.Bound2_kind(W)
        v = W.get_Bound2_kind;
    end
    function set.Bound2_kind(W, v)
        W.set_Bound2(v);
    end

    function v = get.SigmaSq1_kind(W)
        v = W.get_SigmaSq1_kind;
    end
    function set.SigmaSq1_kind(W, v)
        W.set_SigmaSq1(v);
    end

    function v = get.SigmaSq2_kind(W)
        v = W.get_SigmaSq2_kind;
    end
    function set.SigmaSq2_kind(W, v)
        W.set_SigmaSq2(v);
    end

    function v = get_Drift1_kind(W)
        v = W.Drift1.kind;
    end

    function v = get_Drift2_kind(W)
        v = W.Drift2.kind;
    end

    function v = get_Bound1_kind(W)
        v = W.Bound1.kind;
    end

    function v = get_Bound2_kind(W)
        v = W.Bound2.kind;
    end

    function v = get_SigmaSq1_kind(W)
        v = W.SigmaSq1.kind;
    end

    function v = get_SigmaSq2_kind(W)
        v = W.SigmaSq2.kind;
    end
    
    function v = get.Tnd_kind(W)
        v = W.get_Tnd_kind;
    end
    function set.Tnd_kind(W, v)
        W.set_Tnd_kind(v);
    end    
    
    function v = get_Tnd_kind(W)
        v = W.Tnd.distrib;
    end
    function set_Tnd_kind(W, v)
        W.set_Tnd(v);
    end
end
%% Naming / Model setting
methods
    function v = obj2kind(~, objs, obj_kind)
        f = @(c) strrep(bml.pkg.pkg2class(c), obj_kind, '');
        v = {f(objs{1}), f(objs{2})};
        if strcmp(v{1}, v{2})
            v = v{1};
        end
    end
    
    function v = get_drift_kind(W)
        v = {W.Drift1_kind, W.Drift2_kind};
        if strcmp(v{1}, v{2}), v = v{1}; end
    end
    function set_drift_kind(W, v)
        if ischar(v), v = {v, v}; end
        W.Drift1_kind = v{1};
        W.Drift2_kind = v{2};
    end
    
    function v = get_bound_kind(W)
        v = {W.Bound1_kind, W.Bound2_kind};
        if strcmp(v{1}, v{2}), v = v{1}; end
    end
    function set_bound_kind(W, v)
        if ischar(v), v = {v, v}; end
        W.Bound1_kind = v{1};
        W.Bound2_kind = v{2};
    end
    
    function v = get_sigmaSq_kind(W)
        v = {W.SigmaSq1_kind, W.SigmaSq2_kind};
        if strcmp(v{1}, v{2}), v = v{1}; end
    end
    function set_sigmaSq_kind(W, v)
        if ischar(v), v = {v, v}; end
        W.SigmaSq1_kind = v{1};
        W.SigmaSq2_kind = v{2};
    end
    
    function v = get_tnd_distrib(W)
        v = W.get_Tnd_kind;
    end
    function set_tnd_distrib(W, v)
        W.set_Tnd_kind(v);
    end
    
    function v = get_n_tnd(W)
        v = W.Tnd.n_Tnd;
    end
    function v = set_n_tnd(W, v)
        v0 = W.Tnd.n_Tnd;
        if v0 ~= v
            W.Tnd.n_Tnd = v;
            W.Tnd.init_params0;
        end
    end
    
    %% td_kind: D2 specific
    function set.td_kind(W, v)
        W.set_td_kind(v);
    end
    function v = get.td_kind(W)
        v = W.get_td_kind;
    end
    function set_td_kind(W, v)
        W.Dtb.td_kind = v;
    end
    function v = get_td_kind(W)
        v = W.Dtb.td_kind;
    end
end
end