classdef Main ...
        < Fit.Common.Main ...
        & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.Main
    %
    % 2015 YK wrote the initial version.
    
%% Settings - Fit
properties
    % to_use_easiest_only
    % : If +1, calculate cost from the easiest conditions only.
    %   If 0, calculate cost from all conditions.
    %   If -1, calculate cost from non-easiest conditions only.
    %
    % : 1 means the easiest conditions, negative means 'except for'
    %   Note: The convention is opposite to dif_rel_incl, etc.,
    %   where 1 is the hardest condition.
    %
    % : 0 or [] is to use all.
    % : Vector means include all elements.
    %   A nonscalar vector input must contain 
    %   all-positive or all-negative elements.
    to_use_easiest_only = 1;
    
    % to_use_easiest_only to use during fitting
    to_use_easiest_only_for_fit = 1;
    
    % to_use_easiest_only to use during model comparison
    to_use_easiest_only_for_comparison = -1;
    
    % to_include_last_frame
    % : If true, include last frame in calculating cost
    %   If not true, set predicted RT at last frame to zero and normalize
    %   so that p(predicted_RT) sums to 1 within each condition.
    to_include_last_frame = false; % true;
end
%% Properties - common
properties (SetAccess=protected)
    Miss
end
%% Properties - Settings    
properties
    to_use_history = true;
    
%     to_plot = true;
    to_save_plot = true;
    to_plot_kind_ = 'all';    
end
properties (Dependent)
    to_plot_kind
end
properties (Transient)
    W_now % so that when batch is stopped, W can be accessed.
end
%% Init
methods
    function set_Miss(W, obj_or_name)
        default_obj = Fit.Common.Miss;
        if nargin < 2, obj_or_name = W.miss_kind; end
        W.Miss = W.enforce_class(class(default_obj), obj_or_name);
        W.set_sub_from_props({'Miss'});
    end
end
%% Main - Template
methods
    function W = Main(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function [Fls, ress] = batch(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'model_kind', 'RT'
            });
        Ss = bml.args.factorizeS(S_batch);        
        [Fls, ress] = W0.batch_Ss(Ss);
    end
    function [Fls, ress] = batch_Ss(W0, Ss)
        if isstruct(Ss), Ss = num2cell(Ss); end
        n = numel(Ss);
        
        Fls = cell(n, 1);
        ress = cell(n, 1);
        
        for ii = 1:n
            S = Ss{ii};
            C = S2C(S);
            
            W = W0.create(C{:});
%             W = feval(class(W0), C{:});
            W0.W_now = W;
            
            [Fls{ii}, ress{ii}] = W.main;
        end        
    end
    function [files, Ss] = batch_files(W0, varargin)
        S_batch = varargin2S(varargin);
        [Ss, n] = factorizeS(S_batch);
        files = cell(n, 1);
        W = feval(class(W0));
        for ii = 1:n
            S = Ss(ii);
            varargin2props(W, S);
            files{ii} = W.get_file;
        end
    end
    function [Fl, res] = main(W)
        file = W.get_file;
        if W.skip_existing_mat && exist([file, '.mat'], 'file')
            fprintf('Skipping existing fit: %s\n', [file '.mat']);
            if nargout > 0 || (W.to_save_plot && ~W.skip_existing_fig)
                L = load([file '.mat']);
                Fl = W.get_Fl;
                Fl.res = L.res;
                res = L.res;
            else
                Fl = [];
                res = [];
            end
        else
            [Fl, res] = W.fit;
            W.save_mat;
            if W.to_save_plot
                W.plot_and_save_all;
            end
        end
    end
    function save_mat(W)
        if isempty(W.Fl)
            warning('W.Fl is empty! Skipping saving.');
            return;
        elseif isequal(W.Fl.res, struct)
            warning('W.Fl.res is empty! Skipping saving.');
            return;
        end
        
        Fl = W.Fl;
        res = Fl.res;
        L = packStruct(W, Fl, res); %#ok<NASGU>
        
        file = [W.get_file '.mat'];
        mkdir2(fileparts(file));
        
        save(file, '-struct', 'L');
        fprintf('Saved to %s\n', file);        
    end
    function varargout = get_cost_validation(W)
        % [cost, cost_sep] = get_cost_validation(W)
        % : Use to_use_easiest_only_for_comparison for to_use_easiest_only
        to_use_easiest_only0 = W.to_use_easiest_only;
        W.to_use_easiest_only = W.to_use_easiest_only_for_comparison;
        [varargout{1:nargout}] = W.get_cost;
        W.to_use_easiest_only = to_use_easiest_only0;
    end
    function [cost, cost_sep] = calc_cost(W)
        % [cost, cost_sep] = calc_cost(W)
        pred = W.get_pred_pdf;
        data = W.get_data_pdf;
%         pred = W.Data.get_RT_pred_pdf;
%         data = W.Data.get_RT_data_pdf;
        
        if ~W.to_include_last_frame
            % Normalize pred after removing the last frame
            pred = W.set_last_frame_0_and_normalize(pred);
        end
        
        % Reshape into a (cond, bin) matrix for nll_bin.
        siz0 = size(pred);
        n_conds = siz0([2 3]);
        siz  = [prod(siz0([1 4 5])), prod(n_conds)];
        
        pred = reshape(permute(pred, [1 4 5 2 3]), siz);
        data = reshape(permute(data, [1 4 5 2 3]), siz);
        
        [~, cost_sep] = bml.stat.nll_bin( ...
            pred, data, ...
            'normalize', false); % true);
        
        % to_use_easiest_only
        % : 1 means the easiest conditions, negative means 'except for'
        %   Note: The convention is opposite to dif_rel_incl, etc.,
        %   where 1 is the hardest condition.
        %
        % : 0 or [] is to use all.
        % : Vector means include all elements.
        %   A nonscalar vector input must contain 
        %   all-positive or all-negative elements.
        if isempty(W.to_use_easiest_only) ...
            || isequal(W.to_use_easiest_only, 0)
            to_incl = true(n_conds);
        else
            dif_incl = W.to_use_easiest_only;
            if ~(all(dif_incl > 0) || all(dif_incl < 0))
                disp('dif_incl:');
                disp(dif_incl);
                error(['All elements of to_use_easiest_only must have' ...
                       'the same sign']);
            end
            
            to_excl = all(dif_incl < 0);
            if to_excl
                dif_incl = -dif_incl;
            end

            n_dim = 2;
            dif_incls = cell(1, n_dim);

            for dim = 1:n_dim
                conds = uniquetol(W.Data.cond(:,dim));

                % smallest conds = hardest condition becomes 1 in d_cond
                [~,~,d_cond] = unique(abs(conds));

                % Make 1 the easiest condition
                difs = max(d_cond) + 1 - d_cond;
                dif_incls{dim} = ismember(difs, dif_incl);
            end

            to_incl = false(n_conds);
            to_incl(dif_incls{1}, :) = true;
            to_incl(:, dif_incls{2}) = true;
            if to_excl
                to_incl = ~to_incl;
            end
        end
%         disp('Included conditions');
%         disp(to_incl);
        
        cost_sep1 = nansum(cost_sep);
        cost = sum(cost_sep1(to_incl(:)));
        
        if nargout >= 2
            cost_sep = permute(reshape(cost_sep, ...
                siz0([1, 4, 5, 2, 3])), [1, 4, 5, 2, 3]);
        end

%         pred = W.Data.get_RT_pred_pdf;
%         data = W.Data.get_RT_data_pdf;
%         
%         % Normalize within each condition.
%         % Since we observe only RT <= t_max,
%         % we are effectively conditionalizing the RTs.
%         pred = nan0(bsxfun(@rdivide, pred, sums(pred, [1, 4, 5])));
%         
%         % Reshape into a (bin, cond) matrix for nll_bin.
%         siz0 = size(pred);
%         siz  = [prod(siz0([1 4 5])), prod(siz0([2 3]))];
%         
%         [c, c_sep] = bml.stat.nll_bin( ...
%             reshape(permute(pred, [1 4 5 2 3]), siz), ...
%             reshape(permute(data, [1 4 5 2 3]), siz), ...
%             'normalize', false);
%           
% %         [c, ceq] = W.get_constr_res
% 
%         if nargout >= 2
%             c_sep = permute(reshape(c_sep, siz0([1 4 5 2 3])), ...
%                 [1 4 5 2 3]);
%         end

        if any(isnan(pred(:)))
            keyboard;
        end
    end
    function pred = get_pred_pdf(W)
        pred = W.Data.get_RT_pred_pdf;
    end
    function data = get_data_pdf(W)
        data = W.Data.get_RT_data_pdf;
    end
    function p = set_last_frame_0_and_normalize(~, p)
        % p(frame, cond1, cond2, ch1, ch2)
        p(end,:,:,:,:) = 0;
        p = bsxfun(@rdivide, p, sums(p, [1, 4, 5]));
    end
    function [Fl, res] = fit(W, varargin)
        % [Fl, res] = fit(W, varargin)
        %
        % A template for fitting functions.
        % See also: FitFlow.fit_grid
        Fl = W.get_Fl;
        
        S = varargin2S(varargin, {
            'opts', {}
            });
        S.opts = varargin2S(S.opts, {
            'UseParallel', 'always'
            'FiniteDifferenceType', 'central'
            });
        C = S2C(S);
        
        to_use_easiest_only0 = W.to_use_easiest_only;
        W.to_use_easiest_only = W.to_use_easiest_only_for_fit;
        res = Fl.fit(C{:});
        W.to_use_easiest_only = to_use_easiest_only0;
    end
    function Fl = get_Fl(W)
        Fl = W.get_Fl@Fit.D2.Common.CommonWorkspace;        
        Fl.plot_opt.to_plot = W.to_plot;        
    end
end
%% Batch - RT
properties (Constant)
    models_RT = {'Ser', 'Par', 'Exv', 'Trg', 'InhDrift', 'InhFree'}; % , 'InhFixFano'};
    models_RT_Inh = {'Ser', 'Par', 'InhDrift', 'InhFree'}; % , 'InhFixFano'};
    models_RT_long = varargin2S({
        'Ser', 'Serial'
        'Par', 'Parallel'
        'Exv', 'Exhaustive'
        'Trg', 'Targetwise'
        'InhDrift', sprintf('Signal\nSuppression')
        'InhFree', 'Flexible'
        });
end
methods
    %% Collapsing bound w/o changing sigmaSq w/ interaction
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Quad_Ixn(W0, varargin)
        S = varargin2S(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'QuadPreDrift'
            'fix_irr_ixn', false
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Quad_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Quad_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Quad_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Quad_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound w/o changing sigmaSq w/ interaction
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn(W0, varargin)
        S = varargin2S(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'LinearMinPreDrift'
            'fix_irr_ixn', false
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Linear_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Linear_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound and const sigmaSq w/ interaction
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(W0, varargin)
        S = varargin2S(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'Const'
            'fix_irr_ixn', false
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound w/o changing sigmaSq
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Const(W0, varargin)
        S = varargin2C(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'Const'
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Const(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Const(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Collapsing bound and changing sigmaSq
    function S = get_S_batch_fit_RT_Inh_BetaCdf_Linear(W0, varargin)
        S = varargin2C(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'LinearMinPreDrift'
            });
    end
    function batch_fit_RT_Inh_BetaCdf_Linear(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_BetaCdf_Linear(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% Const bound, const sigmaSq, ixn
    function S = get_S_batch_fit_RT_Inh_Const_Const_Ixn(W0, varargin)
        S = varargin2C(varargin, {
            'model', W0.models_RT_Inh
            'bound_kind', 'CosBasis'
            'sigmaSq_kind', 'LinearMinPreDrift'
            });
    end
    function batch_fit_RT_Inh_Const_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_Const_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_fit_RT(C{:});
    end
    function batch_plot_RT_Inh_Const_Const_Ixn(W0, varargin)
        S = W0.get_S_batch_fit_RT_Inh_Const_Const_Ixn(varargin{:});
        C = S2C(S);
        W0.batch_plot_RT(C{:});
    end
    
    %% General
    function S_batch = get_S_batch_RT(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'parad', 'RT'
            'model', W0.models_RT
            });
    end
    function batch_plot_RT(W0, varargin)
        S_batch = W0.get_S_batch_RT(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_RT(C{:});
            
            file = W.get_file;
            L = load(file);
            
            L.Fl.res2W;
            W = L.Fl.W;
            bml.oop.varargin2props(W, C, true);
            W.Fl = L.Fl;
            
            W.plot_and_save_all;
        end
    end
    function varargout = batch_fit_RT(W0, varargin)
        S_batch = W0.get_S_batch_RT(varargin{:});
        Ss = bml.args.factorizeS(S_batch);
        
        [varargout{1:nargout}] = W0.batch_Ss(Ss);
    end
    function Ls = batch_load_RT(W0, varargin)
        % Ls(file): Struct containing:
        % .Fl
        % .res
        
        S_batch = W0.get_S_batch_RT(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        for ii = n:-1:1
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_RT(C{:});
            file = W.get_file;
            fprintf('Loading (%d/%d): %s\n', ii, n, file);
            L1 = load([file, '.mat']);
            Ls(ii) = copyFields(S, L1, {'Fl', 'res'});
        end
    end    
    function W = create(W0, varargin)
        S = varargin2S(varargin, {
            'model_kind', 'RT' % 'RT'|'short'
            });
        C = S2C(S);
        switch S.model_kind
            case 'RT'
                W = W0.create_RT(C{:});
            case 'short'
                W = W0.create_sh(C{:});
        end
    end
    function W = create_RT(~, varargin)
        S = varargin2S(varargin, {
            'model', 'Ser'
            ...
            'drift_kind', 'Const'
            ... 'bound_kind'
            ... : 'Const'|'BetaCdf'|'BetaMeanAsym'|'CosBasis'
            'bound_kind', 'Const' 
            'sigmaSq_kind', 'Const' % 'Const'|'LinearMinPreDrift'
            'fix_irr_ixn', false % true
            'fix_miss', false
            'fix_fano', true
            'fano_max', 1
            });
        
        switch S.model
            case {'Ser', 'Par'}
                S = varargin2S(S, {
                    'td_kind', S.model
                    });
                C = S2C(S);
                
                if S.fix_irr_ixn
                    W = Fit.D2.Bounded.Main(C{:});
                else
                    W = Fit.D2.IrrIxn.Main(C{:});
                end
                
            case 'InhSer'
                S = varargin2S(S, {
                    'drift_fac_1', 0
                    'drift_fac_2', 0
                    'sigmaSq_fac_1', 0
                    'sigmaSq_fac_2', 0
                    'p_dim1_1st', 0
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false
                    'fix_fano_2', false
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
                W.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim1_2 = 0;
                W.th.Dtb__drift_sigmaSq_fac_dim2_1 = 0;
                W.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                W.fix_to_th_({
                    'Dtb__drift_sigmaSq_fac_dim1_1'
                    'Dtb__drift_sigmaSq_fac_dim1_2'
                    'Dtb__drift_sigmaSq_fac_dim2_1'
                    'Dtb__drift_sigmaSq_fac_dim2_2'
                    });
                
            case 'InhPar'
                S = varargin2S(S, {
                    'drift_fac_1', 1
                    'drift_fac_2', 1
                    'sigmaSq_fac_1', 1
                    'sigmaSq_fac_2', 1
                    'p_dim1_1st', 0
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false
                    'fix_fano_2', false
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
                W.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim1_2 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim2_1 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                W.fix_to_th_({
                    'Dtb__drift_sigmaSq_fac_dim1_1'
                    'Dtb__drift_sigmaSq_fac_dim1_2'
                    'Dtb__drift_sigmaSq_fac_dim2_1'
                    'Dtb__drift_sigmaSq_fac_dim2_2'
                    });
                
            case 'InhDrift'
                S = varargin2S(S, {
                    'drift_fac_1', 0
                    'drift_fac_2', 0
                    'sigmaSq_fac_1', 1
                    'sigmaSq_fac_2', 1
                    'p_dim1_1st', 0.5
                    'fix_p_dim1_1st', false
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', true
                    'fix_fano_2', true
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
            case 'InhSlice'
                S = varargin2S(S, {
                    'drift_fac_1', 0.5
                    'drift_fac_2', 0.5
                    'sigmaSq_fac_1', 0.5
                    'sigmaSq_fac_2', 0.5
                    'p_dim1_1st', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false % true
                    'fix_fano_2', false % true
                    'fix_fano', false
                    'dtb', 'DensitySlice'
                    }, S);
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});                
                
            case 'InhSliceFree'
                S = varargin2S(S, {
                    'drift_fac_1', 0.5
                    'drift_fac_2', 0.5
                    'sigmaSq_fac_1', 0.5
                    'sigmaSq_fac_2', 0.5
                    'p_dim1_1st', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false % true
                    'fix_fano_2', false % true
                    'fix_fano', false
                    'dtb', 'DensitySliceFree'
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});                
                
            case 'InhSliceFix'
                S = varargin2S(S, {
                    'drift_fac_1', 0.5
                    'drift_fac_2', 0.5
                    'sigmaSq_fac_1', 0.5
                    'sigmaSq_fac_2', 0.5
                    'p_dim1_1st', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false % true
                    'fix_fano_2', false % true
                    'fix_fano', false
                    'dtb', 'DensitySliceFix'
                    'slprops0', [0.5, 0.5]
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
            
            case 'InhEvScale'
                S = varargin2S(S, {
                    'drift_fac_1', 0.5
                    'drift_fac_2', 0.5
                    'sigmaSq_fac_1', 0.5
                    'sigmaSq_fac_2', 0.5
                    'p_dim1_1st', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false % true
                    'fix_fano_2', false % true
                    'fix_fano', false
                    'dtb', 'DensityEvScale'
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});                
                
            case 'InhFree'
                S = varargin2S(S, {
                    'drift_fac_1', 0
                    'drift_fac_2', 0
                    'sigmaSq_fac_1', 0.16
                    'sigmaSq_fac_2', 0.16
                    'p_dim1_1st', 0.5
                    'fix_p_dim1_1st', false
                    'fix_drift_fac_1', false
                    'fix_drift_fac_2', false
                    'fix_sigmaSq_fac_1', false
                    'fix_sigmaSq_fac_2', false
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', true
                    'fix_fano_2', true
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
%             case 'InhFixFano'
%                 S = varargin2S(S, {
%                     'drift_fac_1', 0.16
%                     'drift_fac_2', 0.16
%                     'sigmaSq_fac_1', 0.16
%                     'sigmaSq_fac_2', 0.16
%                     'p_dim1_1st', 0.5
%                     'fix_p_dim1_1st', false
%                     'fix_drift_fac_1', false
%                     'fix_drift_fac_2', false
%                     'fix_sigmaSq_fac_1', false
%                     'fix_sigmaSq_fac_2', false
%                     'fix_bias_irr_1', S.fix_irr_ixn
%                     'fix_bias_abs_irr_1', S.fix_irr_ixn
%                     'fix_bias_irr_2', S.fix_irr_ixn
%                     'fix_bias_abs_irr_2', S.fix_irr_ixn
%                     'fix_fano_1', true
%                     'fix_fano_2', true
%                     });
%                 C = S2C(S);
%                 W = Fit.D2.Inh.MainBatch(C{:});
            
            case 'Trg'
%                 S = varargin2S(S, S);
                C = S2C(S);
                W = Fit.D2.Targetwise.Main(C{:});
                
            case 'Exv'
%                 S = varargin2S(S, S);
                C = S2C(S);
                W = Fit.D2.RT.Exv.Main(C{:});
                
            otherwise
                error('Unknown model=%s\n', S.model);
        end
    end
end
%% Batch - Short
properties (Constant)
    models_sh = {'Ser', 'Par', 'InhDrift', 'InhFree', 'DurFree'};
end
methods
    function imgather_sh(W0, varargin)
        
    end
    function S_batch = get_S_batch_sh(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_sh
            'parad', 'sh'
            'model', W0.models_sh
            });
    end
    function batch_plot_sh(W0, varargin)
        S_batch = W0.get_S_batch_sh(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_sh(C{:});
            
            file = W.get_file;
            L = load(file);
            
            L.Fl.res2W;
            W = L.Fl.W;
            bml.oop.varargin2props(W, C, true);
            W.Fl = L.Fl;
            
            W.plot_and_save_all;
        end
    end
    function batch_fit_sh(W0, varargin)
        S_batch = W0.get_S_batch_sh(varargin{:});
        [Ss, n] = bml.args.factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            W = W0.create_sh(C{:});
            W0.W_now = W;
            
            W.main;
        end
    end
    function W = create_sh(~, varargin)
        S = varargin2S(varargin, {
            'model', 'Ser'
            'bound', 'Const'
            'fix_irr_ixn', true
            'fix_miss', false
            'fix_sigma', true
            'fix_fano', true
            'fano_max', 1
            ... Short paradigm specific
            'p_dim1_1st', 0.5
            'fix_p_dim1_1st', false
            'buffer_dur_sec', 0.12 - 4/75;
            'fix_drift_t_st', true
            });
        if S.fix_sigma
            S.sigmaSq = 'Const';
        else
            S.sigmaSq = 'LinearPreDrift';
        end
        
        switch S.model
            case {'Ser', 'Par'}
                S = varargin2S(S, {
                    'td_kind', S.model
                    });
                C = S2C(S);
                
                if S.fix_irr_ixn
                    W = Fit.D2.Bounded.Main(C{:});
                else
                    W = Fit.D2.IrrIxn.Main(C{:});
                end
                
            case 'InhSlice'
                S = varargin2S(S, {
                    'drift_fac_1', 0.5
                    'drift_fac_2', 0.5
                    'sigmaSq_fac_1', 0.5
                    'sigmaSq_fac_2', 0.5
                    'p_dim1_1st', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false % true
                    'fix_fano_2', false % true
                    'fix_fano', false
                    'dtb', 'DensitySlice'
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});    
                                
            case 'InhSliceFree'
                S = varargin2S(S, {
                    'drift_fac_1', 0.5
                    'drift_fac_2', 0.5
                    'sigmaSq_fac_1', 0.5
                    'sigmaSq_fac_2', 0.5
                    'p_dim1_1st', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', false % true
                    'fix_fano_2', false % true
                    'fix_fano', false
                    'dtb', 'DensitySliceFree'
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});    
                
                logit_b_mean0 = logit(0.05);
                W.th0.Dtb__Bound1__b_logitmean = logit_b_mean0;
                W.th.Dtb__Bound1__b_logitmean = logit_b_mean0;
                W.th0.Dtb__Bound2__b_logitmean = logit_b_mean0;
                W.th.Dtb__Bound2__b_logitmean = logit_b_mean0;
                
            case 'InhSer'
                S = varargin2S(S, {
                    'drift_fac_1', 0
                    'drift_fac_2', 0
                    'sigmaSq_fac_1', 0
                    'sigmaSq_fac_2', 0
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', true
                    'fix_fano_2', true
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
                W.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim1_2 = 0;
                W.th.Dtb__drift_sigmaSq_fac_dim2_1 = 0;
                W.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                W.fix_to_th_({
                    'Dtb__drift_sigmaSq_fac_dim1_1'
                    'Dtb__drift_sigmaSq_fac_dim1_2'
                    'Dtb__drift_sigmaSq_fac_dim2_1'
                    'Dtb__drift_sigmaSq_fac_dim2_2'
                    });
                
            case 'InhPar'
                S = varargin2S(S, {
                    'drift_fac_1', 1
                    'drift_fac_2', 1
                    'sigmaSq_fac_1', 1
                    'sigmaSq_fac_2', 1
                    'fix_p_dim1_1st', true
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', true
                    'fix_fano_2', true
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
                W.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim1_2 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim2_1 = 1;
                W.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                W.fix_to_th_({
                    'Dtb__drift_sigmaSq_fac_dim1_1'
                    'Dtb__drift_sigmaSq_fac_dim1_2'
                    'Dtb__drift_sigmaSq_fac_dim2_1'
                    'Dtb__drift_sigmaSq_fac_dim2_2'
                    });
                
            case 'InhDrift'
                S = varargin2S(S, {
                    'drift_fac_1', 0
                    'drift_fac_2', 0
                    'sigmaSq_fac_1', 1
                    'sigmaSq_fac_2', 1
                    'p_dim1_1st', 0.5
                    'fix_p_dim1_1st', false
                    'fix_drift_fac_1', true
                    'fix_drift_fac_2', true
                    'fix_sigmaSq_fac_1', true
                    'fix_sigmaSq_fac_2', true
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', true
                    'fix_fano_2', true
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
            case 'InhFree'
                S = varargin2S(S, {
                    'drift_fac_1', 0
                    'drift_fac_2', 0
                    'sigmaSq_fac_1', 0.16
                    'sigmaSq_fac_2', 0.16
                    'p_dim1_1st', 0.5
                    'fix_p_dim1_1st', false
                    'fix_drift_fac_1', false
                    'fix_drift_fac_2', false
                    'fix_sigmaSq_fac_1', false
                    'fix_sigmaSq_fac_2', false
                    'fix_bias_irr_1', S.fix_irr_ixn
                    'fix_bias_abs_irr_1', S.fix_irr_ixn
                    'fix_bias_irr_2', S.fix_irr_ixn
                    'fix_bias_abs_irr_2', S.fix_irr_ixn
                    'fix_fano_1', true
                    'fix_fano_2', true
                    });
                C = S2C(S);
                W = Fit.D2.Inh.MainBatch(C{:});
                
            case 'Trg'
                C = S2C(S);
                W = Fit.D2.Targetwise.Main(C{:});
                
            case 'Exv'
                C = S2C(S);
                W = Fit.D2.RT.Exv.Main(C{:});
        end

        if isfield(W.th0, 'Dtb__Bound1__b_logitmean')
            % Collapsing early for prd=sh
            logit_b_mean0 = logit(0.05);
            W.th0.Dtb__Bound1__b_logitmean = logit_b_mean0;
            W.th.Dtb__Bound1__b_logitmean = logit_b_mean0;
            W.th0.Dtb__Bound2__b_logitmean = logit_b_mean0;
            W.th.Dtb__Bound2__b_logitmean = logit_b_mean0;
        elseif isfield(W.th0, 'Dtb__Dtb1__Bound__b_logitmean')
            % Collapsing early for prd=sh
            logit_b_mean0 = logit(0.05);
            W.th0.Dtb__Dtb1__Bound__b_logitmean = logit_b_mean0;
            W.th.Dtb__Dtb1__Bound__b_logitmean = logit_b_mean0;
            W.th0.Dtb__Dtb2__Bound__b_logitmean = logit_b_mean0;
            W.th.Dtb__Dtb2__Bound__b_logitmean = logit_b_mean0;
        end        
        
        if S.fix_drift_t_st && isa(W, 'Fit.D2.Inh.MainBatch')
            W.th.Dtb__Drift1__log10_t_st = log10(S.buffer_dur_sec);
            W.fix_to_th_('Dtb__Drift1__log10_t_st');
            
            W.th.Dtb__Drift2__log10_t_st = log10(S.buffer_dur_sec);
            W.fix_to_th_('Dtb__Drift2__log10_t_st');
        end
    end
end
%% Table
methods
    function [ds_txt, file, Ls] = tabulate_files(W0, files)
        %%
        n = numel(files);
        Ls = cell(n, 1);
        for ii = 1:n
            file = files{ii};
            if ~exist(file, 'file')
                warning('%s is absent! skipping..\n', file);
                continue;
            end
            L = load(file, 'res', 'W');
            fprintf('Loaded %s\n', file);
            
            % Compute fval & BIC based on the validation set
            W = L.W;
            L.res.fval = W.get_cost_validation;
            
            
            %
            Ls{ii} = L;
        end
        
        %%
        Ss = cell(n, 1);
        for ii = 1:n
            Ss{ii} = Ls{ii}.W.get_S0_file;
        end
        
        %%
        txts = cell(n, 1);
        for ii = 1:numel(files)
            L = Ls{ii};
            if isempty(L)
                continue; 
            end
            
            txt = struct;
            txt = W0.tabulate_metainfo(txt, L);
            txt.file = files{ii};

%             res = L.res;
%             th_names = fieldnames(res.th)';
%             for jj = 1:numel(th_names)
%                 th_name = th_names{jj};
%                 
%                 txt = W0.tabulate_param(txt, th_name, L);
%             end
            txts{ii} = txt;
        end
        ds_txt = bml.ds.from_Ss(txts);
        ds_txt = cell2mat2_ds(ds_txt);
        
        %% Best models
        S0 = Ss{1};
        ic = 'BIC';
        subjs = unique(ds_txt.subj);
        parads = unique(ds_txt.parad);
        
        i_res = 0;
        ds_best = dataset;
        ix_best = [];
        
        for i_subj = 1:numel(subjs)
            subj = subjs{i_subj};
            for i_parad = 1:numel(parads)
                parad = parads{i_parad};
                
                incl = find(strcmp(ds_txt.subj, subj) ...
                    & strcmp(ds_txt.parad, parad));
                [min_ic, min_ix] = min(ds_txt.(ic)(incl));
                min_ix = incl(min_ix);
                
                ds_txt.(['delta_' ic])(incl,1) = ds_txt.(ic)(incl) - min_ic;
                
                i_res = i_res + 1;
                ds_best(i_res,:) = ds_txt(min_ix, :);
                ix_best(i_res) = min_ix; %#ok<AGROW>
            end
        end
        n_res = i_res;
        
        %% Add params to the best models
        txt_best = cell(n_res, 1);
        for i_res = 1:n_res
            ix_best1 = ix_best(i_res);
            txt = struct;
            
            L = Ls{ix_best1};
            res = L.res;
            th_names = fieldnames(res.th)';
            for jj = 1:numel(th_names)
                th_name = th_names{jj};
                
                txt = W0.tabulate_param(txt, th_name, L);
            end
            txt_best{i_res} = txt;
        end
        ds_best_param = bml.ds.from_Ss(txt_best);
        ds_best = [ds_best, ds_best_param];
        
        %%
        S0_fields = fieldnames(Ss{1})';
        for f = S0_fields
            S0.(f{1}) = unique(ds_txt.(f{1}));
        end
        file = W0.get_file_from_S0(S0, {
            'tbl', 'all'
            });
        
        export(ds_txt, 'File', [file '.csv'], 'Delimiter', ';');
        fprintf('Exported fit results to %s.csv\n', file);
        
        %%
        file = W0.get_file_from_S0(S0, {
            'tbl', 'best'
            });
        
        export(ds_best, 'File', [file '.csv'], 'Delimiter', ';');
        fprintf('Exported best fit results to %s.csv\n', file);        
    end
    function txt = tabulate_metainfo(~, txt, L)
        
        S0_file = L.W.get_S0_file;
        txt = copyFields(txt, S0_file);
        
        res = L.res;
        txt.NParam = res.k;
        txt.fval = res.fval;
        txt.BIC = res.bic;
    end
    function txt = tabulate_param(~, txt, th_name, L)
        res = L.res;
        th = res.th.(th_name);
        se = res.se.(th_name);
        if isscalar(th)
            txt.(th_name) = sprintf('%1.3g +- %1.3g', ...
                th, se);
        else
            for ii = 1:numel(th)
                th_name1 = sprintf('%s_%d', th_name, ii);
                txt.(th_name1) = sprintf('%1.3g +- %1.3g', ...
                    th(ii), se(ii));
            end
        end
    end
end
%% Plot - Goodness of Fit
methods
    function axs = imgather_RT_wi_subj(W0, varargin)
        %
        S = W0.get_S_batch_RT(varargin{:});
        S.plot = {
            {
                'plt', 'ch'
                'dX', 1
            }
            {
                'plt', 'ch'
                'dX', 2
            }
            {
                'plt', 'rt'
                'dX', 1
            }
            {
                'plt', 'rt'
                'dX', 2
            }
            };
        n_subj = numel(S.subj);
        n_model = numel(S.model);
        n_plot = numel(S.plot);
        
        %
        for i_subj = n_subj:-1:1
            subj = S.subj{i_subj};
            
            fig_tag(subj);
            clf;
            
            for i_model = n_model:-1:1
                model = S.model{i_model};
                
                W = W0.create_RT('subj', subj, 'model', model);
                
                for i_plot = n_plot:-1:1
                    S_plot = varargin2S(S.plot{i_plot});
                    C_plot = varargin2C( ...
                        copyFields(struct, S_plot, {
                            'plt', 'dX'
                        }));
                    file = [W.get_file(C_plot), '.fig'];
                
                    ax1 = subplotRC(n_model, n_plot, i_model, i_plot);
                    ax1 = bml.plot.openfig_to_axes(file, ax1);
                
                    ax(i_model, i_plot) = ax1;
                end
            end
            axs{i_subj} = ax;
        end
        
        %
        for i_subj = n_subj:-1:1
            subj = S.subj{i_subj};            
            fig_tag(subj);
            ax = axs{i_subj};
            
            for i_model = n_model:-1:1
                model = S.model{i_model};
                
                for i_plot = n_plot:-1:1
                    S_plot = varargin2S(S.plot{i_plot});
                    ax1 = ax(i_model, i_plot);
                    
                    title(ax1, '');

                    if i_plot == 1
                        col_title = W0.models_RT_long.(model);
                        if i_model == n_model
                            ylab = sprintf('%s\nP_{right}', col_title);
                        else
                            ylab = sprintf('%s\n _{ }', col_title);
                        end
                    else
                        if i_model == n_model
                            switch S_plot.plt
                                case 'ch'
                                    if S_plot.dX == 1
                                        ylab = 'P_{right}';
                                    else
                                        ylab = 'P_{blue}';
                                    end
                                case 'rt'
                                    if S_plot.dX == 1
                                        ylab = 'RT (s)';
                                    else
                                        ylab = ' ';
                                    end
                            end
                        else
                            ylab = '';
                        end
                    end
                    ylabel(ax1, ylab);
                    
                    if mod(i_plot, 2) == 0
                        set(ax1, 'YTickLabel', {''});
                    end
                    
                    xticks = cellstr(get(ax1, 'XTickLabel'));
                    if i_model == n_model
                        xticks(2:2:end) = {''};
                    else
                        xticks(:) = {''};
                        xlabel(ax1, '');
                    end
                    set(ax1, 'XTickLabel', xticks);
                    
                    h = bml.plot.children2struct(ax1);
                    set(h.marker, 'MarkerSize', 4);
                    set(h.marker, 'LineWidth', 0.25);
                    set(ax1, 'FontSize', 9);
                end
            end
            
            bml.plot.position_subplots(ax, ...
                'margin_top', 0.02, ...
                'margin_left', 0.125, ...
                'margin_bottom', 0.06, ...
                'btw_row', 0.015, ...
                'btw_col', [0.05, 0.075, 0.05]);
            
            %
            file = W0.get_file({
                'sbj', subj
                'mdl', S.model
                'plt', 'ch_rt'
                });
            bml.plot.savefigs(file, ...
                'PaperPosition', [0, 0, ...
                    Fig.Consts.width_column2_cm, ...
                    n_model * 3.5], ...
                'ext', {'.fig', '.png', '.tif'}); % [600, n_subj * 400]);
        end
    end
    function plot_gof_wi_subj(W0, Ls, varargin)
        S = varargin2S(varargin, {
            'gof', 'bic'
            'gof_label', ''
            });
        
        subjs_all = {Ls.subj};
        subjs = unique(subjs_all);
        n_subj = numel(subjs);
        for i_subj = 1:n_subj
            subj = subjs{i_subj};
    
            clf;
            W0.plot_gof_wi_subj_unit(Ls, subj, varargin{:});
            
            file = W0.get_file({
                'sbj', subj
                'plt', 'gof'
                'gof', S.gof});
            Fig.savefigs_column2(file, n_model * 3.5);
        end
    end
    function imgather_gof(W0, Ls, varargin)
        if iscell(Ls)
            Ls0 = Ls;
            clear Ls
            n = numel(Ls0);
            for f = {'subj', 'parad', 'model', 'Fl', 'res'}
                for ii = n:-1:1
                    Ls(ii).(f{1}) = Ls0{ii}.(f{1});
                end
            end
        end
        
        S = varargin2S(varargin, {
            'gof', 'bic'
            'gof_label', ''
            });
        
        subjs_all = {Ls.subj};
        subjs = unique(subjs_all);
        n_subj = numel(subjs);
        
        clf;
        for i_subj = n_subj:-1:1
            subj = subjs{i_subj};
    
            ax1 = subplotRC(1, n_subj, 1, i_subj);
            [~, dgof{i_subj}] = ...
                W0.plot_gof_wi_subj_unit(Ls, subj, varargin{:});
            
            ax(1, i_subj) = ax1;
        end
        
        bml.plot.position_subplots(ax, ...
            'margin_top', 0.18, ...
            'margin_left', 0.08, ...
            'margin_right', 0.04, ...
            'margin_bottom', 0.22);
        
        for i_subj = 1:n_subj
            ax1 = ax(1, i_subj);
            
            set(ax1, 'FontSize', 9, 'TickLength', [0.015, 0.01]);
            
            if i_subj > 1
                set(ax1, 'YTick', []);
            end
            if i_subj ~= round((n_subj + 1) / 2)
                xlabel(ax1, '');
            end
            
            dgof_sorted = sort(dgof{i_subj});
            
            bnd = 5;
            
            xlim_small_max = min( ...
                ceil(dgof_sorted(end-bnd) / 10) * 12, ...
                dgof_sorted(end) - 1);
            xlim_large_min = max( ...
                floor(dgof_sorted(end) / 100) * 50, ...
                dgof_sorted(end-1) + 1);

            [h, hch] = bml.plot.break_axis(ax1, 'x', ...
                xlim_small_max, xlim_large_min);
            
            if i_subj == round((n_subj + 1) / 2)
                hlabel = hch.label;
                hlabelpos = get(hlabel, 'Position');
                hlabelylim = get(h(3), 'YLim');
                hlabelpos(2) = ...
                    (hlabelpos(2) - hlabelylim(1)) * 1.07 ...
                    + hlabelylim(1);
                set(hlabel, 'Position', hlabelpos);
%                 str_label = get(hlabel, 'String');
%                 delete(hlabel);
%                 xlabel(h(3), sprintf('\n%s', str_label));
            end
            
            xtick = get(h(1), 'XTick');
            set(h(1), 'XTick', xtick([1, end]));
%             set(h(1), 'XTick', xtick([1, max(end - 1, 2)]));
            
            xtick = get(h(2), 'XTick');
            set(h(2), 'XTick', xtick([1, end]));
%             set(h(2), 'XTick', xtick([min(2, end - 1), end]));
        end
        
        file = W0.get_file({
            'sbj', subjs
            'plt', 'gof'
            'gof', S.gof});
        Fig.savefigs_column2(file, 5);
    end
    function [h, dgof] = plot_gof_wi_subj_unit(W0, Ls, subj, varargin)
        S = varargin2S(varargin, {
            'gof', 'bic'
            'gof_label', ''
            });        
        if isempty(S.gof_label)
            S.gof_label = upper(S.gof);
        end
        
        ax1 = gca;
        
        subjs_all = {Ls.subj};
        incl = strcmp(subj, subjs_all);
        Ls1 = Ls(incl);

        models = {Ls1.model};

        n_models = numel(Ls1);
        gof = zeros(n_models, 1);
        for i_model = 1:n_models
            L = Ls1(i_model);
            gof(i_model) = L.res.(S.gof);
        end

        dgof = gof - min(gof);
        h = barh(1:n_models, dgof, 'k');
        set(ax1, 'YTickLabel', models);
        set(ax1, 'YDir', 'reverse');
        
        xlabel(['\Delta ' S.gof_label]);
        bml.plot.beautify;

        title(sprintf('Subject %s\n', subj(1)));        
    end
end
%% PlotFun
methods
    function add_plotfun(W, Fl)
        W.add_plotfun@Fit.D2.Common.CommonWorkspace(Fl);
        Fit.D2.Common.Plot.PlotFuns.add_plotfun(Fl);
    end
end
%% Comparison plot
methods
    function plot_cost_distrib_all_diff_models(W0, ...
            batch_args, model_args)
        
        if ~exist('batch_args', 'var')
            batch_args = {};
        end
        S_batch = varargin2S(batch_args, {
            'subj', {'DX', 'MA', 'VL'}
            });
        [Ss, n_batch] = bml.args.factorizeS(S_batch);
        
        if ~exist('model_args', 'var')
            model_args = {
                W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn('model', 'Ser')
                W0.get_S_batch_fit_RT_Inh_BetaCdf_Linear_Ixn('model', 'Par')
                };
        end
        n_model = numel(model_args);
        
        for i_batch = 1:n_batch
            Ls = cell(1, n_model);
            for i_model = 1:n_model
                S = Ss(i_batch);
                S_model = varargin2S(model_args{i_model});
                C = varargin2C(S, S_model);
                
                W = W0.create_RT(C{:});
                Ls{i_model} = load(W.get_file);
                
                if i_model >= 2
                    W = Ls{1}.Fl.W;
                    W2 = Ls{i_model}.Fl.W;
                    
                    clf;
                    W.plot_cost_distrib_all_diff(W2);
                    file = W.get_file({
                        'plt', 'cstdf'
                        'mdl2', S_model.model
                        });
                    savefigs(file, 'size', [900, 900]);
                end
            end
        end
    end
end
%% Plot
methods
    function plot_and_save_all(W, varargin)
        S = varargin2S(varargin, {
            'conds_oversample_factor', 1 % 10
            'subdir', class(W)
            });
        
        if is_in_parallel
            warning('Cannot plot when in parallel pool!');
            return;
        end
        
        for dimOnX = 1:2
            W.Data.set_conds_oversample_factor( ...
                1, ...
                dimOnX);
        end

        is_estimated = false;
        
        is_to_plot = @(v) any(ismember(vVec(v), W.to_plot_kind)) ...
            || isequal(W.to_plot_kind, 'all');
        
        % Plots without oversample
        kinds = {
                'plotfuns',  'plotfuns', {'size', [1200, 800]}
                'rt_distrib_all', 'rtdst', {'size', [800, 800]}
                };
        for kind = kinds'
            [kind_long, kind_short, savefigs_args] = deal(kind{:});
            
            if ~is_to_plot({kind_long, kind_short})
                continue;
            end
            
            file = W.get_file({'plt', kind_short}, {}, S.subdir);

            if exist([file, '.fig'], 'file') && W.skip_existing_fig
                fprintf('Skipping existing figure: %s\n', [file, '.fig']);
            else
                if ~is_estimated
                    W.Fl.res2W;
                    is_estimated = true;
                end
                
                try
                    clf;
                    W.(['plot_' kind_long]);
                    savefigs(file, savefigs_args{:});
                catch err
                    warning(err_msg(err));
                end
            end
        end
        
        % Plots with oversample
        kinds = {
            'ch',        'ch'
            'rt',        'rt'
            ... 'rt_stdev',  'rtsd'
            ... 'rt_skew',   'rtsk'
            };
        
        if is_to_plot(kinds(:,1:2))
            for dimOnX = 1:2
                W.Data.set_conds_oversample_factor( ...
                    S.conds_oversample_factor, ...
                    dimOnX);
                W.pred;

                for kind = kinds'

                    [kind_long, kind_short] = deal(kind{:});

                    if ~is_to_plot({kind_long, kind_short})
                        continue;
                    end

                    file = W.get_file({
                                'plt', kind_short, 'dX', dimOnX}, ...
                                {}, S.subdir);
                    if exist([file, '.fig'], 'file') && W.skip_existing_fig
                        fprintf('Skipping existing figure: %s\n', ...
                            [file, '.fig']);
                    else
                        try
                            clf;
                            W.(['plot_' kind_long])('dimOnX', dimOnX);
                            title(sprintf('%s-dimOnX=%d', ...
                                strrep(kind_long, '_', '-'), ...
                                dimOnX));
                            savefigs(file);
                        catch err
                            warning(err_msg(err));
                        end
                    end
                end
            end
        end
    end
    function plot_plotfuns(W)
        W.get_Fl;
        W.Fl.runPlotFcns;
    end
    function varargout = plot_ch(W, varargin)
        Pl = DtbPlot.PlotCh2D;
        [varargout{1:nargout}] = Pl.plot_W_pred_data(W, varargin{:});
    end
    function varargout = plot_rt(W, varargin)
        Pl = DtbPlot.PlotRt2D;
        [varargout{1:nargout}] = Pl.plot_W_pred_data(W, varargin{:});
    end
    function varargout = plot_rt_wrong(W, varargin)
        C = varargin2C(varargin, {
            'accuOnlyAxis', [2, 0]
            });
        Pl = DtbPlot.PlotRt2D;
        [varargout{1:nargout}] = Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_rt_distrib_all(W, varargin)
        C = varargin2C(varargin, {
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_rt_distrib_all_data(W, varargin)
        C = varargin2C(varargin, {
            'src', {'data'}
            'use_bias', false
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_rt_distrib_all_pred(W, varargin)
        % [hd, hp, Pl_d, Pl_p] = plot_rt_distrib_all_pred(W, ...)
        C = varargin2C(varargin, {
            'src', {'pred'}
            'accuOnlyAxis', [0, 0]
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W_pred_data(W, C{:});
    end
    function varargout = plot_cost_distrib_all(W, varargin)
        C = varargin2C(varargin, {
            'src', 'cost'
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W(W, C{:});
    end
    function varargout = plot_cost_distrib_all_diff(W, W2, varargin)
        C = varargin2C(varargin, {
            'src', 'cost_dif'
            'W2', W2
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W(W, C{:});
    end
    function varargout = plot_cost_distrib_all_diff_cum(W, W2, varargin)
        C = varargin2C(varargin, {
            'src', 'cost_dif_cum'
            'W2', W2
            });
        Pl = DtbPlot.PlotRtDistribAll2D;
        [varargout{1:nargout}] = ...
            Pl.plot_W(W, C{:});
    end
    function varargout = plot_rt_d1_vs_d2(W, varargin)
        S = varargin2S(varargin, {
            'abs_cond', true
            'fun', 'mean'
            'yfun', [] % @mean
            'efun', [] % @sem
            });
        switch S.fun
            case 'mean'
                S.yfun = @mean;
                S.efun = @sem;
            case 'var'
                S.yfun = @var;
                S.efun = @sev;
            otherwise
                error('Unknown fun=%s\n', S.fun);
        end
        
        %%
        cond = W.Data.ds.cond;
        rt = W.Data.ds.RT;
        
        %%
        if S.abs_cond
            cond = abs(cond);
        end
        n_dim = size(cond, 2);
        n_tr = size(cond, 1);
        d_cond = zeros(n_tr, n_dim);
        for dim = n_dim:-1:1
            [~,~,d_cond(:,dim)] = unique(cond(:,dim));
        end
        
        %%
        y = accumarray(d_cond, rt, [], S.yfun, nan);
        e = accumarray(d_cond, rt, [], S.efun, nan);
        
        %%
%         y = bsxfun(@minus, y, y(:,1));
        
        %%
        n_line = size(y, 2);
        colors = hsv2rev(n_line);
        
        for i_line = 1:n_line
            errorbar(y(:,end), y(:,i_line), e(:,i_line), ...
                'o-', 'Color', colors(i_line,:));
            hold on;
        end
        hold off;
        axis equal
        grid on;
        bml.plot.beautify;
        
        xlabel(S.fun);
        ylabel(S.fun);
        title(W.get_title);
    end
    function varargout = plot_rt_mean_vs_var(W, varargin)
        S = varargin2S(varargin, {
            'abs_cond', true
            'xfun', @mean
            'yfun', @var
            'efun', @sev
            });
        
        %%
        cond = W.Data.ds.cond;
        rt = W.Data.ds.RT;
        
        %%
        if S.abs_cond
            cond = abs(cond);
        end
        n_dim = size(cond, 2);
        n_tr = size(cond, 1);
        d_cond = zeros(n_tr, n_dim);
        for dim = n_dim:-1:1
            [~,~,d_cond(:,dim)] = unique(cond(:,dim));
        end
        
        %%
        x = accumarray(d_cond, rt, [], S.xfun, nan);
        y = accumarray(d_cond, rt, [], S.yfun, nan);
        e = accumarray(d_cond, rt, [], S.efun, nan);
        
        %%
%         y = bsxfun(@minus, y, y(:,1));
        
        %%
        n_line = size(y, 2);
        colors = hsv2rev(n_line);
        
        for i_line = 1:n_line
            errorbar(x(:,i_line), y(:,i_line), e(:,i_line), ...
                'o-', 'Color', colors(i_line,:));
            hold on;
        end
        hold off;
        axis equal
        
    end
    function set.to_plot_kind(W, v)
        if ischar(v) || ~strcmp(v, 'all')
            v = {v};
        else
            assert(iscell(v) && all(cellfun(@ischar, v(:))));
        end
        W.to_plot_kind_ = v;
    end
    function v = get.to_plot_kind(W)
        v = W.to_plot_kind_;
    end
end
%% Bias
properties
    cond_bias
end
methods
    function b = get.cond_bias(W)
        % b(dim) : bias. Used in plotting, etc.
        b = W.get_cond_bias;
    end
    function b = get_cond_bias(~)
        warning('Implement in subclasses!');
        b = {0, 0};
    end
end
%% File
methods
    function v = get_file_fields0(W)
        v = union_general( ...
                W.get_file_fields0@Fit.Common.Main, ...
                W.get_file_fields0@Fit.D2.Common.CommonWorkspace, ...
            'stable', 'rows');            
        v = union_general(v, ...
            {
            'to_use_easiest_only_for_fit', 'ef'
            'to_use_easiest_only_for_comparison', 'ec'
            'to_include_last_frame', 'lf'
            }, 'stable', 'rows');            
    end
end
%% Demo
methods
    function demo_params(W)
        %% Effect of inhibition of drift
        W.plot_rt;
        
        %% Effect of inhibition of diffusion
        
    end
end
end