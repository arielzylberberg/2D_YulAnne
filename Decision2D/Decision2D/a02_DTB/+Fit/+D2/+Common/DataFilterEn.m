classdef DataFilterEn < Fit.D2.Common.CommonWorkspace
properties
    %% Time
    
    refresh_rate = 75;
    
    smooth_width_sec = 0.05;
    
    d_fr_dim = 0; % latency_dim_2 - latency_dim_1 % TODO
    d_fr_incl = -40:40;
    d_fr_kind = 'rel_minus_irr'; % 'rel_minus_irr'|'irr_minus_rel'
    
    truncate_first_msec = -inf; % Previously -inf.
    truncate_last_msec = 0; % Previously -inf which included tapering end.
    
    tr_prop_thres = 0.9; % was 0 before introducing this param
    tr_n_thres = 20;
    
    % t_plot_max:
    % Defaults to longer than longest 10 percentile RT across subj
    t_plot_max = 1.2; 
    
    %% Time filters
    
    t0_kind = 'st'; % 'st'|'en' % Applied on loading En
end
properties (Transient)
    ens = {}; % ens{feat}(tr,fr)
    ens_cell = {}; % ens{feat}{tr,1}(1,fr)
    ch = []; % ch(tr,1)
end
%% User interface
methods
    function W = DataFilterEn(varargin)
        W.dif_rel_incl = 'all'; % 1:3;
        W.dif_irr_incl = 'all'; % 1:3;
        W.accu_rel_incl = [0 1];
        W.accu_irr_incl = [0 1];    

        W.set_dt;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        bml.oop.varargin2props(W, varargin, true);
        W.set_dt;
        W.init@Fit.D2.Common.CommonWorkspace;
    end
end
%% Loading data
methods
    function load_data(W)
        W.set_path;
        
        if W.Data.loaded
            fprintf('%s already loaded - skipping loading.\n', ...
                W.Data.get_path);
        else
            W.ens = {}; % The only difference from Fit.D2.Common.CommonWorkspace
            W.Data.load_data;
            fprintf('Done.\n');
            
            W.filt_data;
        end
    end
end
methods (Static)
    function W = loadobj(W)
        % Backward compatibility
        if isstruct(W) && ...
                ~isfield(W, 'rt_incl_unit') ...
                || (isfield(W, 'rt_incl_unit') && isempty(W.rt_incl_unit))
            if ~isempty(W.rt_incl_ms)
                W.rt_incl_unit = 'ms';
                W.rt_incl_ = W.rt_incl_ms;
            elseif ~isempty(W.rt_incl_prct)
                W.rt_incl_unit = 'prct';
                W.rt_incl_ = W.rt_incl_prct;
            else
                W.rt_incl_unit = 'prct';
                W.rt_incl_ = [0 100];
            end
        end
    end
end
methods
    function v = get_file_fields0(W)
        v = union_general({
            't0_kind', 't0'
            'truncate_first_msec', 'tst' % 'trc_s'
            'truncate_last_msec', 'ten' % 'trc_e'
            'smooth_width_sec', 'smth'
%             'subj', 'sbj'
%             'parad', 'prd'
%             'n_dim_task', 'nd_tsk'
%             'task', 'tsk'
%             'dim_rel_W', 'dim_r'
%             'dif_irr_incl', 'dif_i'
%             'accu_irr_incl', 'acc_i'
%             'dif_rel_incl', 'dif_r'
            }, W.get_file_fields0@Fit.D2.Common.CommonWorkspace, ...
            'stable', 'rows');
    end
    function v = get_file_mult(W)
        v = {
            'smooth_width_sec', 1000
            };
    end
    function set_Data(W, Data)
        if nargin < 2 || isempty(Data)
            Data = 'En';
        end
        W.set_Data@Fit.D2.Common.CommonWorkspace(Data);
    end
end
%% En time
methods
    function reset_ens(W)
        W.ens_cell = [];
        W.ens = [];
    end
    function ens_cell = get_ens_cell(W)
        persistent t0_kind_prev
        
        if ~isempty(W.ens) && isequal(t0_kind_prev, W.t0_kind)
            ens_cell = W.ens_cell;
            return;
        end
        
        switch W.t0_kind
            case 'st'
                ens_cell = W.Data.get_ens_cell({
                    'truncate_first_sec', W.truncate_first_msec / 1000
                    'truncate_last_sec', W.truncate_last_msec / 1000
                    });
            case 'en'
                ens_cell = W.Data.get_ens_cell({
                    't0', W.Data.get_RT0
                    'to_flip_time', true
                    'truncate_first_sec', W.truncate_first_msec / 1000
                    'truncate_last_sec', W.truncate_last_msec / 1000
                    });
        end

%         n_dim = W.Data.get_n_dim;
%         
%         fr_incl_max = length(W.get_t_plot); % round(W.t_plot_max * W.refresh_rate) + 1;
%         for dim = 1:n_dim
%             ens{dim} = ens{dim}(:, 1:fr_incl_max);
%         end        
        
        % Cache
        t0_kind_prev = W.t0_kind;
        W.ens_cell = ens_cell;
    end
    function ens = get_ens_mat(W)
        persistent t0_kind_prev
        
        % ens_mat{dim} ~= cell2mat2(ens_cell{dim}) because
        % dim(ens_mat, 2) is always fr_max, not max(n_fr)
        
        if ~isempty(W.ens) && isequal(t0_kind_prev, W.t0_kind)
            ens = W.ens;
            return;
        end
        
        st_args = {
            'truncate_first_sec', W.truncate_first_msec / 1000
            'truncate_last_sec', W.truncate_last_msec / 1000
            };
        en_args = {
            't0', W.Data.get_RT0
            'to_flip_time', true
            'truncate_first_sec', W.truncate_first_msec / 1000
            'truncate_last_sec', W.truncate_last_msec / 1000
            };
        
        switch W.t0_kind
            case 'st'
                ens = W.Data.get_ens_mat(st_args);
            case 'en'
                ens = W.Data.get_ens_mat(en_args);
                
            case 'se' % st for dim1, en for dim2
                ens{1} = W.Data.get_en_mat(1, st_args);
                ens{2} = W.Data.get_en_mat(2, en_args);
                
            case 'es' % en for dim1, st for dim1
                ens{1} = W.Data.get_en_mat(1, en_args);
                ens{2} = W.Data.get_en_mat(2, st_args);
        end

%         n_dim = W.Data.get_n_dim;
%         
%         fr_incl_max = length(W.get_t_plot); % round(W.t_plot_max * W.refresh_rate) + 1;
%         for dim = 1:n_dim
%             ens{dim} = ens{dim}(:, 1:fr_incl_max);
%         end        
        
        % Cache
        t0_kind_prev = W.t0_kind;
        W.ens = ens;
    end
    function en = get_en_mat_dim_rel(W)
        en = W.get_en_mat_dim(W.get_dim_rel_W);
    end
    function en = get_en_mat_dim_irr(W)
        en = W.get_en_mat_dim(W.get_dim_irr_W);
    end
    function en = get_en_mat_dim(W, dim)
        ens = W.get_ens_mat;
        en = ens{dim};
    end
    function dur = get_dur_fr_dim(W, dim)
        en = W.get_en_mat_dim(dim);
        n_fr = size(en, 2);
        dur = n_fr - sum(isnan(en), 2);
    end
    function rt = get_rt_fr(W)
        rt = W.Data.get_RT_ix;
    end
    function v = get_smooth_width_fr(W)
        v = W.smooth_width_sec * W.refresh_rate;
    end
    function set_dt(W, varargin)
        if isempty(varargin)
            W.set_dt(1 / W.refresh_rate);
        else
            W.set_dt@Fit.D2.Common.CommonWorkspace(varargin{:});
        end
    end
end
%% Export
methods
    function files1 = batch_export_data(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            'parad', 'RT'
            'dif_rel_incl', 'all'
            'dif_irr_incl', 'all'
            });
        [Ss, n] = factorizeS(S_batch);
        
        files1 = cell(n, 1);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            W = feval(class(W0), C{:});
            files1{ii} = W.export_data;
        end
    end
    function file = export_data(W0, file0)
        if nargin >= 2
            if iscell(file0)
                for file1 = file0(:)
                    W0.export_data(file1{1});
                end
                return;
            end

            %%
            L0 = load(file0);
            if isfield(L0, 'W')
                W = L0.W;
            else
                W = L0.Fl.W;
            end
            W.init;
            
%             L0.Fl.res2W;
%             W = L0.Fl.W;
        else
            W = W0;
            file0 = W.get_file;
        end
        
        %% Copy contents
        L = struct;
%         L.Td_pred_pdf_tr = W.Data.Td_pred_pdf_tr;
%         L.RT_pred_pdf_tr = W.Data.RT_pred_pdf_tr;
        L.ch = W.Data.ch;
        L.cond = W.Data.cond;
%         L.th = W.th;
        L.ens_mat = W.Data.get_ens_mat;
%         L.en_rel_mat = W.Data.en_rel_mat;
        L.rt_fr = W.Data.convert_RT_sec2fr_ix(W.Data.ds.RT); %#ok<STRNU>
        
        %% Save
        [pth, nam] = fileparts(file0);
        file = fullfile(pth, [nam, '+exp=data.mat']);
        mkdir2(fileparts(file));
        save(file, '-struct', 'L');
        fprintf('Data exported to %s\n', file);
    end
end
end