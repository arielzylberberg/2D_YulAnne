classdef Main < Fit.D2.Common.CommonWorkspace
    % Fit.D2.IrrSep.Main
    % 
    % Fit each dim separately for each irr dif.
    
    % 2016 (c) Yul Kang. hk2699 at columbia dot edu.
    
%% Settings
properties
    group_kind = 'dif_irr_incl';
    group_val = num2cell(1:5);
    class_Ws = 'Fit.D1.Bounded.Main';
end
%% Internal
properties % (Transient)
    Ws = {}; % {group} = W for 1D Dtb for the group.
    S0_files_ = {}; % {group}
    files_ = {}; % {group}
end
properties (Dependent)
    n_group % refers to group_val
    group_val_orig % original value of the coherence
    group_kind_orig % Motion/Color Strength
    
    S0_file0 % S0_file that does not refer to Ws{1}
    S0_files % {group} % refers to Ws.
    files % {group} = refers to Ws.
    
    class_Ws_name % e.g., 'Fit^D1^Bounded^Main' for file name
end
%% Results
properties (Dependent)
    ress % (group)
end
%% Batch
methods
    function batch_all(W0, varargin)        
        % When giving cell array values, enclose in one more curly braces.
        %
        % Will be superseded by Fit.main_dtb_irrsep
        
        Cs = W0.get_C_batch_all(varargin{:});
        
        for ii = 1:numel(Cs)
            C = Cs{ii};
            W0.batch(C{:});
        end
    end
    function Cs = get_C_batch_all(~, varargin)
        % Will be superseded by Fit.main_dtb_irrsep
        
        Cs = {};
%         %% 1D
%         C_data = {
%                 {
%                 'n_dim_task', 1
%                 'dim_rel_W', 1
%                 }
%                 {
%                 'n_dim_task', 1
%                 'dim_rel_W', 2
%                 }
%             };
%         C_group = {
%                 {
%                 'group_kind', 'dif_irr_incl'
%                 'group_val', {num2cell(1:2:5)}
%                 'accu_irr_incl', 1
%                 }
%                 {
%                 'group_kind', 'cond_irr_incl'
%                 'group_val', {num2cell(1:2:9)}
%                 'accu_irr_incl', 1
%                 }
%             };
%         Cs = [Cs; bml.args.factorize_merge_C({C_data, C_group})];
        %% 2D
        C_data = {
                {
                'n_dim_task', 2
                'dim_rel_W', 1
                }
                {
                'n_dim_task', 2
                'dim_rel_W', 2
                }
            };
        C_group = {
                {
                'group_kind', 'dif_irr_incl'
                'group_val', {num2cell(1:5)}
                'accu_irr_incl', 1
                }
%                 {
%                 'group_kind', 'cond_irr_incl'
%                 'group_val', {num2cell(1:9)}
%                 'accu_irr_incl', 1
%                 }
                {
                'group_kind', 'dif_irr_incl'
                'group_val', {num2cell(1:3)} % Only difficult irr has accu=0
                'accu_irr_incl', 0
                }
%                 {
%                 'group_kind', 'cond_irr_incl'
%                 'group_val', {num2cell(3:7)} % Only difficult irr has accu=0
%                 'accu_irr_incl', 0
%                 }
            };
        Cs = [Cs; bml.args.factorize_merge_C({C_data, C_group})];
        %% model
        C_model = {
% %                 {
% %                 'class_Ws', 'Fit.D1.MeanRt.Main'
% %                 'fix_miss', true
% %                 'fix_bias_st', true
% %                 }
%                 {
%                 'class_Ws', 'Fit.D1.MeanRt.Main'
%                 'fix_miss', false
%                 'fix_bias_st', true
%                 'kbratio_kind', 'bk'
%                 }
% %                 {
% %                 'class_Ws', 'Fit.D1.Bounded.Main'
% %                 'bound_kind', 'CosBasis'
% %                 'sigmaSq_kind', 'Const'
% %                 'fix_miss', false
% %                 'fix_bias_st', false
% %                 }
                {
                'class_Ws', 'Fit.D1.Bounded.Main'
                'bound_kind', 'CosBasis'
                'sigmaSq_kind', 'LinearMinPreDrift'
                'fix_miss', false
                'fix_bias_st', false
                'fix_sigmaSq_st', false
                }
                {
                'class_Ws', 'Fit.D1.Bounded.Main'
                'bound_kind', 'BetaMeanAsym'
                'sigmaSq_kind', 'LinearMinPreDrift'
                'fix_miss', false
                'fix_bias_st', false
                'fix_sigmaSq_st', false
                }
            };        
        %% combine
        Cs = bml.args.factorize_merge_C( ...
            {Cs, C_model}, ...
            varargin);
    end
    function files = ls_files(W0, filt, criteria)
        %%
        filt = fullfile('Data', class(W0), '*.mat');
        criteria = varargin2C(varargin, {
            'allof', {
                'clw=Fit^D1^MeanRt^Main'
                'msf=0'
                'grpk=cond_irr_incl'
                'prd=RT'
                'aci=1'
                }
            });
        
        S2s = bml.str.Serializer;
        files = S2s.ls(filt, criteria{:});
        
        %%
        W0.batch_plot_files(files);
    end
    function batch_plot_files(W0, files)
        if nargin < 2
            files = bml.str.clipboard2list;
        end
        n = numel(files);
        
        for ii = 1:n
            file = files{ii};
            
            W = feval(class(W0));
            W = W.load_mat(file);
            
            W.plot_and_save_all;
        end
    end
%     function batch_imgather(W0, varargin)
%         Cs = W0.get_C_batch_all(varargin{:});
%         W0.imgather_plots(Cs{1});
%     end
end
%% Main
methods
    function W = Main(varargin)
        W.n_dim_task = 2;
        W.accu_irr_incl = 1;
        
        W.add_deep_copy({'Ws'});
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W0, varargin)
        W0.init@Fit.D2.Common.CommonWorkspace(varargin{:});

        S_batch = varargin2S({
            W0.group_kind, W0.group_val
            });
        [Ss, n] = factorizeS(S_batch);
        
        for ii = n:-1:1
            C = varargin2C(Ss(ii), varargin2S(varargin, W0.S0_file0));
            W0.Ws{ii} = feval(W0.class_Ws, C{:});
        end
    end
    function main(W0)
        if ~W0.is_initialized, W0.init; end
        
        n = W0.n_group;
        file = [W0.get_file, '.mat'];
        if ~exist(file, 'file') || ~W0.skip_existing_mat
            for ii = 1:n
                W = W0.Ws{ii};
                file = W.get_file;
                if exist([file, '.mat'], 'file') ...
                        && W0.skip_existing_mat
                    W.load_mat;
                else
                    if isempty(W.Data.ds)
                        th_vec = W.th_vec;
                        n_th = length(th_vec);
                        W.th_vec = nan(size(th_vec));
                        W.get_Fl;
                        W.Fl.res.out.x = W.th_vec;
                        W.Fl.res.th = W.th;
                        W.Fl.res.se = W.th;
                        W.Fl.res.out.hessian = nan(n_th, n_th);
                        W.save_mat;
                    else                        
                        W.main;
                    end
                end
            end        
            W0.save_mat;
        end
        if W0.to_plot
            W0.plot_and_save_all;
        end
    end
    function fit(W0, varargin)
        for ii = 1:W0.n_group
            W0.Ws{ii}.fit(varargin{:});
        end
    end
    function pred(W0)
        for ii = 1:W0.n_group
            W0.Ws{ii}.pred;
        end
    end
    function save_mat(W, file)
        if ~exist('file', 'var')
            file = W.get_file;
        end
        
        L = W.get_struct_for_saving; %#ok<NASGU>
        mkdir2(fileparts(file));
        save([file, '.mat'], '-struct', 'L');
        fprintf('Saved to %s.mat\n', file);
    end
    function L = get_struct_for_saving(W)
        L = copyFields(struct, W, {
            'S0_file'
            'class_Ws'
            'group_kind'
            'group_val'
            'S0_files'
            'ress'});
        L.W = W; % For convenience
    end
    function [W0, L] = load_mat(W0, file)
        if nargin < 2
            file = [W0.get_file, '.mat'];
        end
        L = load(file);
        
        if isfield(L, 'W')
            W0 = L.W;
        else
            C = S2C(rmfield(L.S0_file, {
                'dif_rel_incl_name'
                'dif_irr_incl_name'
                'cond_irr_incl_name'
                'accu_irr_incl_name'
                }));
            if nargout > 0
                W0 = feval(class(W0), C{:});
            else
                W0.init(C{:});
            end
        end
        
        n = numel(L.group_val);
        for ii = 1:n
            W = W0.Ws{ii};
            if isa(W, 'Fit.D1.MeanRt.Main') ...
                    && ~isempty(fieldnames(W.children))
                for child = fieldnames(W.children)'
                    W.remove_child(child{1});
                end
            end

            W0.Ws{ii}.get_Fl;
            W0.Ws{ii}.Fl.res = L.ress{ii};
            W0.Ws{ii}.Fl.res2W;
        end
    end
    function fs = get_file_fields0(W)
        fs = union_general( ...
            W.get_file_fields0@Fit.D2.Common.CommonWorkspace, ...
            {
            'class_Ws_name', 'clw'
            'group_kind', 'grpk'
            'group_val',  'grpv'
            }, 'stable', 'rows');
        if ~isempty(W.Ws) ...
                && strcmp(class(W.Ws{1}), W.class_Ws)
            fs = union_general(fs, ...
                W.Ws{1}.get_file_fields, ...
                'stable', 'rows');
        end
    end
    function S0_file0 = get.S0_file0(W)
        S0_file0 = W.get_S0_file(false);
    end
    function S0_file = get_S0_file(W, refer_to_Ws)
        if nargin < 2
            refer_to_Ws = true; 
        end
        S0_file = W.get_S0_file@Fit.D2.Common.CommonWorkspace;
        if refer_to_Ws ...
                && ~isempty(W.Ws) ...
                && strcmp(class(W.Ws{1}), W.class_Ws)
            S0_file = copyFields(S0_file, W.Ws{1}.S0_file);
        end
    end
    function v = get.class_Ws_name(W)
        v = strrep(W.class_Ws, '.', '^');
    end
    function set.class_Ws_name(W, v)
        W.class_Ws = strrep(v, '^', '.');
    end
end
%% Plot - batch
methods
    function plot_and_save_all(W0)
%         kinds = {'rt', 'ch', 'bound'};
%         kinds = {'rt', 'ch', 'bound', 'params', 'param_vs_params'};
        kinds = {'params', 'bound'};
%         kinds = {'rt', 'ch', 'bound', 'params', ...
%             'params_vs_RT_easiest_rel_by_irr'};
%         kinds = {'bound'};
        for kind = kinds
            try
                W0.(['plot_' kind{1}]);
            catch err
                warning(err_msg(err));
                continue;
            end                
        end
    end
    function batch_all_imgather_files(W0)
        disp('filt_row:');
                    %%
        filt = fullfile('../Data', class(W0), '*.fig');
        files0 = dirfiles(filt);
        %%
        C_data = {
                {
                'prd', 'RT'
                'dtk', 2
                }
                {
                'prd', 'RT'
                'dtk', 1
                }
%                 {
%                 'prd', 'sh'
%                 'dtk', 2
%                 }
%                 {
%                 'prd', 'sh'
%                 'dtk', 1
%                 }
            };
        C_group = {
%                 {
%                 'grpk', 'dif_irr_incl'
%                 'aci', '[0,1]'
%                 }
                {
                'grpk', 'cond_irr_incl'
                'aci', '[0,1]'
                }
%                 {
%                 'prd', 'RT'
%                 'grpk', 'dif_irr_incl'
%                 'aci', 0
%                 }
%                 {
%                 'prd', 'RT'
%                 'grpk', 'cond_irr_incl'
%                 'aci', 0
%                 }
            };
        C_model = {
% %                 {
% %                 'clw', 'Fit^D1^MeanRt^Main'
% %                 'msf', 1
% %                 'bnd', 'C'
% %                 'ssq', 'C'
% %                 }
%                 {
%                 'clw', 'Fit^D1^MeanRt^Main'
%                 'msf', 0
%                 'bnd', 'C'
%                 'ssq', 'C'
%                 'kb', 'bk'
%                 }
% %                 {
% %                 'clw', 'Fit^D1^Bounded^Main'
% %                 'msf', 0
% %                 'bnd', 'O'
% %                 'ssq', 'C'
% %                 'fbst', 0
% %                 }
%                 {
%                 'clw', 'Fit^D1^Bounded^Main'
%                 'msf', 0
%                 'bnd', 'O'
%                 'ssq', 'C' % 'LMPrD'
%                 'fbst', 0
%                 }
                {
                'clw', 'Fit^D1^Bounded^Main'
                'msf', 0
                'bnd', 'A'
                'ssq', 'C' % 'LMPrD'
                'fbst', 0
                'tnd', 'i'
                'fsqs', 0
                }
            };
        [Cs, n] = bml.args.factorize_merge_C({C_data, C_group, C_model});
        for ii = 1:n
            Cs{ii} = varargin2S(Cs{ii});
        end
        
        %%
        for ii = 1:n
            W0.batch_imgather_files(files0, 'allof', Cs{ii});
        end
        
        %%
        for ii = 1:n
            W0.batch_imgather_files_params(files0, 'allof', Cs{ii});
        end
    end
    function batch_imgather_files(W0, files0, varargin)
        %%
        S = varargin2S(varargin, {
            'allof', {} % filter
            'kinds', {
%                 {'plt', 'rt'}
%                 {'plt', 'ch'}
                {'plt', 'bound'}
                }
            });
        
        kinds = S.kinds;
        n_kind = numel(kinds);
        
        % Sort to ensure the same file name for the same subset,
        % since the combined file name uses the first file name
        % with a postfix.
        files0 = sort(files0); 
        
        S2s = bml.str.Serializer;       
        %%
        for ii = 1:n_kind
            kind = varargin2S([ ...
                bml.args.S2C2(kinds{ii})
                bml.args.S2C2(S.allof)
                ]);
            kind = strsplit(S2s.convert(kind), '+');
            
            files = S2s.ls(files0, ...
                'allof', kind, ...
                'noneof', {'plts=imgather'});
            
            if isempty(files)
                warning('No files that meet criteria: %s\n', ...
                    sprintf('\n%s', kind{:}));
                continue;
            end
            
            axs = W0.imgather_files_subj_dim(files);
            
            n_row = size(axs, 1);
            n_col = size(axs, 2);
            
            is_ch = any(ismember(kind, {'plt=ch'}));
            is_rt = any(ismember(kind, {'plt=rt'}));
            is_rt_or_ch = is_rt || is_ch;
            
            %%
            for row = 1:n_row
                for col = 1:n_col
                    ax1 = axs(row, col);
                    
                    xy = bml.plot.get_all_xy;
                    if isempty(xy)
                        continue;
                    end
                    
                    if is_rt_or_ch
                        Fit.Plot.beautify_coh_axis(ax1);
                        
                        hs = bml.plot.figure2struct(ax1);
                        set(hs.marker, 'MarkerSize', 6, 'LineWidth', 0.05);
                        set(hs.line, 'LineWidth', 1.5);
                    end
                    if is_ch
                        Fit.Plot.beautify_ch_axis(ax1);
                    end
                    
                    if col == 1
                        if row < n_row
                            ax1.YLabel.String = sprintf('S%d\n \n ', row);
                        else
                            y_label = ax1.YLabel.String;
                            if ischar(y_label)
                                ax1.YLabel.String = ...
                                    sprintf('S%d\n\n%s', row, y_label);
                            end
                        end
                    else
                        if is_ch
                            if row < n_row
                                ax1.YLabel.String = ' ';
                            end
                        else
                            ax1.YLabel.String = ' ';
                        end
                        set(ax1, 'YTickLabel', ...
                            repmat({''}, [1, numel(ax1.YTick)]));
                    end
                    if (row < n_row) % || (col > 1)
                        ax1.XLabel.String = ' ';
%                         set(ax1, 'XTickLabel', []);
                    end
                    title(ax1, '');
                end
                sameAxes(axs(row,:), [], [], 'y');
                
                for col = 1:n_col
                    ax1 = axs(row, col);
                    if is_rt
                        Fit.Plot.beautify_rt_axis(ax1, 'lim_from', 'lim');
                        if col > 1
                            ax1.YLabel.String = ' ';
                        end
                    end
                end
            end
            
            if is_rt_or_ch
                bml.plot.position_subplots(axs, ...
                    'margin_left', 0.15, ...
                    'btw_row', 0.1, ...
                    'margin_top', 0.02, ...
                    'margin_bottom', 0.1);
            end
            set(axs, 'FontSize', 9);
            
            file0 = files{1};
            [pth, nam] = fileparts(file0);
            file1 = fullfile(pth, [nam '+plts=imgather']);
            savefigs(file1);
        end
    end
    function batch_imgather_files_params_compare_all(W0, files_imgather0, varargin)
        S = varargin2S(varargin, {
            'filt', {}
            });
        S.filt = varargin2S(S.filt, {
            'clw', 'Fit^D1^MeanRt^Main'
            'grpk', 'dif_irr_incl' % 'cond_irr_incl' % 
%             'aci', 1
            'dtk', 2
            'msf', 0
            'kb', 'bk'
            'th', {
                'tnd_mu'
                'tnd_sd'
                'b_experienced'
                'k'
                'kb'
% %                 'kbratio'
                'bkratio'
                'bias_cond'
                'miss'
                }
            });
        [filts, n] = factorizeS(S.filt);

        if ~exist('files_imgather0', 'var') ...
                || isempty(files_imgather0)
            S2s = bml.str.Serializer;
            files_imgather0 = S2s.ls( ...
                fullfile('Data', class(W0), '*.fig'), ...
                'allof', 'plts=imgather');
        end
        
        S2s = bml.str.Serializer;
        for ii = 1:n
            filt = filts(ii);
            filt_C = strsplit(S2s.convert(filt), '+');
            
            W0.batch_imgather_files_params_compare(files_imgather0, ...
                'filt', filt_C, ...
                varargin{:});
        end
    end
    function batch_imgather_files_params_compare(W0, files_imgather0, varargin)
        S = varargin2S(varargin, {
            'filt', {}
            'compare', {'aci=1', 'aci=0'} % {'dtk=1', 'dtk=2'}
            'color', {'k', 'r'}
            'jitter', 0.04
            });
        n_compare = numel(S.compare);
        jitter = ((1:n_compare) - (1 + n_compare) / 2) * S.jitter;
        files_imgather0 = sort(files_imgather0);
        
        %%
        files = cell(n_compare, 1);
        S2s = bml.str.Serializer;
        for ii = 1:n_compare
            files1 = S2s.ls(files_imgather0, ...
                'allof', [S.filt(:); S.compare(ii)], ...
                'noneof', {'cmp={aci=1,aci=0}', 'cmp={dtk=1,dtk=2}'});
            assert(isscalar(files1));
            files{ii} = files1{1};
        end
        
        %%
        fig = figure(1);
        clf(fig);
        axs = cell(n_compare,1);
        
        for ii = n_compare:-1:1
            figs(ii) = openfig(files{ii}, 'invisible');
            axs{ii} = bml.plot.subplot_by_pos(figs(ii));
        end
        
        copyobj(axs{1}, fig);
        axs0 = bml.plot.subplot_by_pos(fig);
        n_row = size(axs0, 1);
        n_col = size(axs0, 2);
        for row = 1:n_row
            for col = 1:n_col
                delete(get(axs0(row, col), 'Children'));
            end
        end
        
        %%
        for ii = 1:n_compare
            for row = 1:n_row
                for col = 1:n_col
                    ax1 = axs{ii}(row, col);
                    children = get(ax1, 'Children');
                    
                    color = S.color{ii};
                    
                    set(children, ...
                        'Color', color, ...
                        'MarkerFaceColor', color);
                end
            end
        end
        
        %%
        for row = 1:n_row
            for col = 1:n_col
                ax_dst = axs0(row, col);
                
                for comp = 1:n_compare
                    ax_src = axs{comp}(row, col);
                    children0 = get(ax_src, 'Children');
                    children = copyobj(children0, ax_dst);
                    axis(ax_dst, 'auto');
                    
                    for ii = 1:numel(children)
                        child = children(ii);
                        set(child, 'XData', ...
                            get(child, 'XData') + jitter(comp));
                    end
                end
                
                bml.plot.beautify_lim('ax', ax_dst, 'lim_from', 'markers')
            end
        end
        
        %%
        for comp = 2:n_compare
            delete(figs(comp));
        end
        
        %%
        S2s = bml.str.Serializer;
        [pth, nam] = fileparts(files{1});
        file_out = fullfile(pth, ...
            [nam '+cmp=' S2s.convert(S.compare)]);

        bml.plot.position_subplots(axs0, ...
            'margin_left', 0.17, ...
            'margin_right', 0.02, ...
            'margin_top', 0.07, ...
            'btw_col', 0.08);
        savefigs(file_out, 'h_fig', fig);
    end
    function batch_imgather_files_params(W0, files0, varargin)
        S = varargin2S(varargin, {
            'allof', {} % filter
            'kinds', {
                'k'
                'kb'
                'kbratio'
                'bkratio'
                'bias_cond'
%                 'b'
                'bias_bound'
                'miss'
                'tnd_mu'
                'tnd_disper'
                'log10_ssq_max_cond'
                ...
                'b_experienced'
                'ssq_max_cond'
                'tnd_sd'
                'ssq_st'
                };
            });
        
        kinds = S.kinds;
        n_kind = numel(kinds);
        
        % Sort to ensure the same file name for the same subset,
        % since the combined file name uses the first file name
        % with a postfix.
        files0 = sort(files0); 
        
        S2s = bml.str.Serializer;
        
        for ii = 1:n_kind
            kind = {
                'plt', 'th_rt'
                'th', kinds{ii}
                };
            S_kind = varargin2S([ ...
                bml.args.S2C2(kind)
                bml.args.S2C2(S.allof)
                ]);
            kind = strsplit(S2s.convert(S_kind), '+');
            
            files = S2s.ls(files0, ...
                'allof', kind, ...
                'noneof', {'plts=imgather'});
            
            if isempty(files)
                warning('No files for kind=%s\n', ...
                    S2s.convert(kind));
                disp(S_kind);
                continue;
            end
            axs = W0.imgather_files_subj_dim(files);
            
            n_row = size(axs, 1);
            n_col = size(axs, 2);
            
            for row = 1:n_row
                for col = 1:n_col
                    ax1 = axs(row, col);
                    
                    if col == 1
                        if row < n_row
                            ax1.YLabel.String = sprintf('S%d\n \n ', row);
                        else
                            y_label = ax1.YLabel.String;
                            ax1.YLabel.String = ...
                                sprintf('S%d\n\n%s', row, y_label);
                        end
                    else
                        ax1.YLabel.String = ' ';
                    end
                    if row == 1
                        title(ax1, Data.Consts.dimNames_long{col});
                    end
                    if (row < n_row) % || (col > 1)
                        ax1.XLabel.String = ' ';
                        set(ax1, 'XTickLabel', []);
                    end
                end
            end
            
            file0 = files{1};
            [pth, nam] = fileparts(file0);
            file1 = fullfile(pth, [nam '+plts=imgather']);
            savefigs(file1);
        end
    end
    function axs = imgather_files_subj_dim(W0, files0, varargin)
        S = varargin2S(varargin, {
            'row_args', csprintf('sbj=%s', Data.Consts.subjs_RT)
            'col_args', csprintf('dmr=%d', 1:2)
            'excl', {'dfr=all'} % exclude old version
            });
        
        n_row = numel(S.row_args);
        n_col = numel(S.col_args);
                
        S2s = bml.str.Serializer;
        
        clf;
        axs = subplotRCs(n_row, n_col);
        
        for row = 1:n_row
            for col = 1:n_col
                filt_row = S.row_args(row);
                filt_col = S.col_args(col);
                
                files1 = S2s.ls(files0, ...
                    'allof', [filt_row(:); filt_col(:)], ...
                    'noneof', S.excl);
                
%                 ix = ~cellfun(@isempty, strfind(files0, filt_row)) ...
%                     & ~cellfun(@isempty, strfind(files0, filt_col)) ...
%                     & cellfun(@isempty, strfind(files0, S.excl));
% 
%                 if nnz(ix) == 0
%                     continue;
%                 end
%                 assert(nnz(ix) == 1);
%                 file = files0{ix};

                if ~isscalar(files1)
                    warning('~isscalar(files1)!');
                    disp('files1:');
                    disp(files1);
                    disp('filt_row:');
                    disp(filt_row);
                    disp('filt_col:');
                    disp(filt_col);
                    disp('S.excl:');
                    disp(S.excl);
                    continue;
                end
                file = files1{1};
                
                ax1 = axs(row, col);
                axs(row, col) = openfig_to_axes(file, ax1);
            end
        end
    end
end
%% Plot - imgather - direct (doesn't work)
methods
    function imgather_plots(W0, varargin)
        warning('Doesn''t work yet');
        
        C = varargin2C(varargin);
        
        for kind = {
                'ch', 'rt', 'bound'
                }
            W0.imgather_subj_dim(C, {
                'plt', kind{1}
                });
        end
    end
    function varargout = imgather_subj_dim(W0, page_args, add_args, varargin)
        warning('Doesn''t work yet');
        
        if nargin < 2, page_args = {}; end
        if nargin < 3, add_args = {}; end
        
        S_page = bml.struct.rmfield(varargin2S(page_args, {
            'n_dim_rel', 2
            'parad', 'RT'
            }), {'dim_rel_W', 'subj'});
        S_page = factorizeS(S_page);
        assert(isscalar(S_page));
        
        page_args = varargin2C(S_page);
        
        col_args = varargin2C({
            'dim_rel_W', {1, 2}
            });
        row_args = varargin2C({
            'subj', Data.Consts.(['subjs_' S_page.parad])
            });
        
        [varargout{1:nargout}] = W0.imgather_page( ...
            row_args, col_args, page_args, add_args, varargin{:});
    end
end
%% Plot - parameters
methods
    function plot_tnd_vs_mean_rt(W0)
        % Caveat: mean RT may not match tnd when bound differs between 
        % irr_dif.
        
        jitter = 0.01
        
        [h, res] = W0.plot_param('tnd_mu');
        
        n_group = W0.n_group;
        for i_group = n_group:-1:1
            W = W0.Ws{i_group};
            mean_rt(i_group) = mean(W.Data.rt);
            sem_rt(i_group) = sem(W.Data.rt);
        end
        
        cla;
        mean_rt1 = mean_rt - min(mean_rt);
        errorbar_wo_tick(res.x - jitter, mean_rt1, sem_rt, [], {
            'LineStyle', 'none'
            }, {
            'LineStyle', '-'
            });
        hold on;

        y = res.y - min(res.y);
        errorbar_wo_tick(res.x + jitter, y, res.le, res.ue, {
            'Color', 'r'
            });
        hold off;
        bml.plot.beautify_lim;
    end
    function plot_params(W0)
%         for th_name = fieldnames(W0.Ws{1}.Fl.res.th)'
        for th_name = {
%                 'k'
%                 'bias_cond'
%                 'kb'
%                 'kbratio'
%                 'bkratio'
% %                 'b'
                'b_experienced'
%                 'bias_bound'
%                 'miss'
%                 'tnd_mu'
%                 'tnd_sd'
% %                 'tnd_disper'
%                 'ssq_max_cond'
% %                 'log10_ssq_max_cond'
%                 'ssq_st'
                }'
            clf;
            
            try
                W0.plot_param(th_name{1});
            catch err
                warning(err_msg(err));
                continue;
            end
            
            file = W0.get_file({
                'plt', 'th'
                'th', th_name{1}
                });
            savefigs(file);
        end
    end
    function [h, res] = plot_param(W0, th_name)
        x = mean(cell2mat2(W0.group_val), 2);
        [y, lb, ub] = W0.get_th_Ws(th_name);
        
        le = lb - y;
        ue = ub - y;
        [h.marker, h.error] = errorbar_wo_tick(x, y, le, ue, {
            'Marker', 'o'
            'MarkerFaceColor', 'k'
            'MarkerEdgeColor', 'w'
            'LineStyle', 'none'
            }, {
            });
        
        y_range = max(y(:)) - min(y(:)) + eps;
        ylim([min(y(:)) - y_range / 2, max(y(:)) + y_range / 2]);
        
        ylabel(strrep(th_name, '_', '-'));
        xlabel(strrep(W0.group_kind, '_', '-'));
        set(gca, ...
            'XTick', x, ...
            'XTickLabel', Fit.Plot.beautify_coh_labels(W0.group_val_orig));
        xlim([0.5, (max(x) + 0.5)]);
        bml.plot.beautify;
        Fit.Plot.xlabel(gca, ...
            'feat', Data.Consts.dimNames{W0.dim_irr_W});
        
        res = packStruct(x, y, lb, ub, le, ue);
    end
    function plot_params_vs_RT_easiest_rel_by_irr(W0, varargin)
        for th_name = {
                'k'
%                 'bias_cond'
%                 'kb'
%                 'kbratio'
%                 'bkratio'
% %                 'b'
%                 'b_experienced'
%                 'bias_bound'
%                 'miss'
%                 'tnd_mu'
%                 'tnd_sd'
% %                 'tnd_disper'
%                 'ssq_max_cond'
% %                 'log10_ssq_max_cond'
%                 'ssq_st'
                }'
            clf;
            
            try
                W0.plot_param_vs_RT_easiest_rel_by_irr(th_name{1});
            catch err
                warning(err_msg(err));
                continue;
            end
            
            file = W0.get_file({
                'plt', 'th_rt'
                'th', th_name{1}
                });
            savefigs(file);
        end        
    end
    function plot_param_vs_RT_easiest_rel_by_irr(W0, th_name)
        x = W0.get_RT_easiest_rel_by_irr;
        [y, lb, ub] = W0.get_th_Ws(th_name);
        
        le = lb - y;
        ue = ub - y;
        [h.marker, h.error] = errorbar_wo_tick(x, y, le, ue, {
            'Marker', 'o'
            'MarkerFaceColor', 'k'
            'MarkerEdgeColor', 'w'
            'LineStyle', 'none'
            }, {
            });
        
        y_range = max(y(:)) - min(y(:)) + eps;
        ylim([min(y(:)) - y_range / 2, max(y(:)) + y_range / 2]);
        
        ylabel(strrep(th_name, '_', '-'));
        xlabel('RT (s)');
        bml.plot.beautify;
        
        res = packStruct(x, y, lb, ub, le, ue);
    end
    function x = get_RT_easiest_rel_by_irr(W0)
        n = W0.n_group;
        x = zeros(n, 1);
        
        for ii = 1:n
            W = W0.Ws{ii};
            p = mean([ ...
                W.Data.RT_data_pdf(:,1,1), ...
                W.Data.RT_data_pdf(:,end,2)], 2);
            
            x(ii) = mean_distrib(p, W.t(:));
        end
    end
    function plot_param_vs_params(W0)
        ths = {
            'k', 'b'
            'k', 'bias_cond'
            'tnd_mu', 'k' % tnd_mu: mean tnd_mu
            'tnd_mu', 'bias_cond'
            'tnd_mu', 'b'
            'tnd_mu', 'bias_bound'
            'tnd_mu', 'miss'
            }';
        for th = ths
            [th1, th2] = deal(th{:});
            clf;
            try
                W0.plot_param_vs_param(th1, th2);
            catch err
                warning(err_msg(err));
                continue;
            end
                
            file = W0.get_file({
                'plt', 'th_th'
                'th1', th1
                'th2', th2
                });
            savefigs(file);
        end
    end
    function [hs, res] = plot_param_vs_param(W0, th_name1, th_name2)
        [x, xlb, xub] = W0.get_th_Ws(th_name1);
        [y, ylb, yub] = W0.get_th_Ws(th_name2);
        
        xle = xlb - x;
        xue = xub - x;
        yle = ylb - y;
        yue = yub - y;
        
        n = W0.n_group;
        colors = hsv2(n);
        
        for ii = n:-1:1
            [h.marker, h.err_x, h.err_y] = bml.plot.errorbar_wo_tick2( ...
                x(ii), y(ii), xle(ii), xue(ii), yle(ii), yue(ii), {
                'LineStyle', 'none'
                'Marker', 'o'
                'MarkerFaceColor', colors(ii,:)
                'MarkerEdgeColor', 'w'
                }, {
                'LineStyle', '-'
                'Marker', 'none'
                'Color', colors(ii,:)
                });
            hold on;
            hs{ii} = h;
        end        
        hold off;
        
        x_range = max(x(:)) - min(x(:)) + 1e-6;
        xlim([min(x(:)) - x_range / 2, max(x(:)) + x_range / 2]);
        
        y_range = max(y(:)) - min(y(:)) + 1e-6;
        ylim([min(y(:)) - y_range / 2, max(y(:)) + y_range / 2]);
        
        bml.plot.beautify;
        xlabel(strrep(th_name1, '_', '-'));
        ylabel(strrep(th_name2, '_', '-'));
        
        res = packStruct(x, y, xlb, ylb, xub, yub);
    end
    function [y, lb, ub] = get_th_Ws(W0, th_name)
        [y, lb, ub] = cellfun(@(W) W.get_res_param(th_name), W0.Ws, ...
            'UniformOutput', false);
        y = cell2mat2(y);
        lb = cell2mat2(lb);
        ub = cell2mat2(ub);

%         y = cellfun(@(W) W.Fl.res.th.(th_name), W0.Ws, ...
%             'UniformOutput', false);
%         y = cell2mat2(y);
        
%         if nargout >= 2
%             e = cellfun(@(W) W.Fl.res.se.(th_name), W0.Ws, ...
%                 'UniformOutput', false);
%             e = cell2mat2(e);
%         end
    end
end
%% Plot - RT, Ch
methods
    function plot_rt(W0)
        n = W0.n_group;
        colors = hsv2rev(n);
        clf;
        for ii = 1:n
            color = colors(ii,:);
            W = W0.Ws{ii};
%             if isempty(W.Fl) || ~W.Fl.is_valid_res
%                 continue;
%             end
            [h_data, h_pred] = W.plot_rt('to_plot_wrong', false);
            hold on;
            for ch = 1:numel(h_data.correct)
                set(h_data.correct(ch), 'Color', color, 'MarkerFaceColor', color);
                set(h_pred.correct(ch), 'Color', color, 'MarkerFaceColor', color);
            end
            
            if isfield(h_data, 'correct_err')
                for jj = 1:numel(h_data.correct_err)
                    delete(h_data.correct_err{jj});
                end
            end
        end
        hold off;
        bml.plot.beautify;

        Fit.Plot.beautify_rt_axis;
        Fit.Plot.beautify_coh_axis;
        
        file = W0.get_file({'plt', 'rt'});
        savefigs(file);
    end
    function plot_ch(W0)
        n = W0.n_group;
        colors = hsv2rev(n);
        clf;
        for ii = 1:n
            color = colors(ii,:);
            W = W0.Ws{ii};
%             if isempty(W.Fl) || ~W.Fl.is_valid_res
%                 continue;
%             end
            [h_data, h_pred] = W.plot_ch;
            hold on;
            hs = bml.plot.figure2struct;
            delete(hs.segment_vert);
            for ch = 1:numel(h_data)
                set(h_data(ch), 'Color', color, 'MarkerFaceColor', color);
                set(h_pred(ch), 'Color', color, 'MarkerFaceColor', color);
            end
        end
        hold off;
        bml.plot.beautify;

        Fit.Plot.beautify_ch_axis;
        Fit.Plot.beautify_coh_axis;

        file = W0.get_file({'plt', 'ch'});
        savefigs(file);
    end
    function plot_bound(W0)
        n = W0.n_group;
        colors = hsv2rev(n);
        clf;
        for ii = 1:n
            color = colors(ii,:);
            W = W0.Ws{ii};
            if isempty(W.Fl) || ~W.Fl.is_valid_res
                continue;
            end
            [h_line, h_err] = W.Dtb.Bound.plot_w_se;
            h_line = [h_line{:}];
            h_err = [h_err{:}];
            set(h_line, 'Color', color);
            set(h_err, 'FaceColor', color);
            hold on;
        end
        bml.plot.beautify;
        xlabel('Time (s)');
        ylabel(sprintf('%s Bound (a.u.)', ...
            Data.Consts.dimNames_long{W.dim_rel_W}));
        
        file = W0.get_file({'plt', 'bound'});
        savefigs(file);
    end
end
%% Get/Set utilities
methods
    function v = get.n_group(W0)
        v = numel(W0.group_val);
    end
    function v = get.group_val_orig(W0)
        switch W0.group_kind
            case 'cond_irr_incl'
                v = W0.Data.conds0_wo_oversample_all_dim{W0.dim_irr_W};
                v = v(cell2mat(W0.group_val));
                
            case 'dif_irr_incl'
                v0 = W0.Data.conds0_wo_oversample_all_dim{W0.dim_irr_W};
                v0 = unique(abs(v0));
                v = v0(cell2mat(W0.group_val));
                
            otherwise
                error('Unsupported group_kind=%s\n', W.group_kind);
        end
    end
    function S0_files = get.S0_files(W0)
        S0_files = cellfun(@(W) W.S0_file, W0.Ws, ...
            'UniformOutput', false);
    end
    function files = get.files(W0)
        files = cellfun(@(W) W.get_file, W0.Ws, ...
            'UniformOutput', false);
    end
    function ress = get.ress(W0)
        if isempty(W0.Ws)
            warning('Ws is empty: ress will be empty, too!');
            ress = [];
            return;
        end
        
        ress = cell(size(W0.Ws));
        for ii = 1:numel(W0.Ws)
            W = W0.Ws{ii};
            if isempty(W.Fl)
                warning('Ws{%d}.Fl is empty! ress{%d} will be empty.', ...
                    ii, ii);
                ress{ii} = [];
            else
                ress{ii} = W.Fl.res;
            end
        end
    end
end
end