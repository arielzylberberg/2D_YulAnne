classdef MainLags < EnIxn.GLM.MainGLM
%% Settings - Optional - Inherited
properties
%     lev_kind = 'beta'; % 'beta'|'bsame'|'bsameonly'
%     lev_cum = []; % []: false; true
%     thres_plot_prop = 0.9;
%     
%     res_props = {'est', 'ci', 'n_tr_in_fr', 'n_tr_total'};
end
%% Settings - Optional
properties
    t_st_reg_ms = 0;
    t_en_reg_ms = 1000;
    
    t0_rel = 'st';
    t0_opp = false;
    
    smooth_ms = 100;
    
    seed = 0;
    n_shuf = 200;
end
%% Internal
properties (Dependent)
    t_st_reg_fr
    t_en_reg_fr
    t_reg % Time vector of the regressors
    t0_irr % Same as t0_rel if t0_opp = false;
    t0_kinds % {t0_dim1, t0_dim2}
end
%% Results - Inherited
properties
    est0 % original estimate
    ests % {shuf}(fr,fr)
    
%     est % (fr, fr) % inherited
%     ci % (fr, fr, [lb, ub]) % inherited
%     n_tr_in_fr % (fr, fr) % inherited
%     n_tr_total % scalar
end
%% Init
methods
    function W = MainLags(varargin)
        W.lev_kind = 'beta'; % 'beta';
        W.t0_kind = 'st';
        W.res_props = union(W.res_props, {'est0'}); % , 'ests'});
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Batch
methods
    function batch(W0, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            ... 'dif_incl', {1, 1:2}
            't0_rel', {'st'}
            });
        [Ss, n] = factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            if ~isfield(S, 'dif_rel_incl')
                S.dif_rel_incl = S.dif_incl;
            end
            if ~isfield(S, 'dif_irr_incl')
                S.dif_irr_incl = S.dif_incl;
            end
            
            C = S2C(S);
            W = feval(class(W0), C{:});
            W0.W_now = W;
            
            W.main;
        end
    end
end
%% Main
methods
    function main(W)
        W.calculate;
        W.plot_and_save_all;
    end
end
%% Calculate
methods
    function calculate(W)
        %% Load results if existing
        [loaded, file] = W.load_if_existing;
        if loaded
            return;
        end
        
        %% Init input
        dim_rel_W = W.dim_rel_W;

        ens_cell = W.get_ens_cell;
        ch = W.Data.ch == 2;
        cond = W.Data.cond;
        
        if dim_rel_W == 2
            ens_cell = ens_cell([2, 1]);
            ch = ch(:, [2, 1]);
            cond = cond(:, [2, 1]);
        end
        
        ens_mat = cell(1, 2);        
        t0_kinds = {W.t0_rel, W.t0_irr};
        for dim = 1:2
            switch t0_kinds{dim}
                case 'st'
                    % Do nothing
                case 'en'
                    % Flip
                    ens_cell{dim} = ...
                        cellfun(@flip, ens_cell{dim}, ...
                            'UniformOutput', false);
                otherwise
                    error('Unknown t0_kinds{%d}=%s\n', ...
                        dim, t0_kinds{dim});                    
            end
            ens_mat{dim} = cell2mat2(ens_cell{dim});
        end
        
        t_st_reg_fr = W.Time.convert_sec2fr_ix(W.t_st_reg_ms / 1e3);
        t_en_reg_fr = W.Time.convert_sec2fr_ix(W.t_en_reg_ms / 1e3);
        reg_fr = t_st_reg_fr:t_en_reg_fr;
        smooth_fr = round(W.smooth_ms / 1e3 / W.dt);        
        
        lev_kind = W.lev_kind;

        for dim = 1:2
            ens_mat{dim} = ens_mat{dim}(:, reg_fr);
        end
        
        
        n_tr = size(ens_mat{1}, 1);
        n_shuf = W.n_shuf;
        seed = W.seed;
        
        %% Process
        rng(seed);
        ix_shuf = cell(1, n_shuf);
        for dim = 2:-1:1
            [~,~,group{dim}] = unique([cond(:,dim), ch(:,dim)], 'rows');
        end
        for i_shuf = 1:n_shuf
            ix_shuf{i_shuf,1} = vVec(bml.stat.randperm_group(group{1}));
            ix_shuf{i_shuf,2} = vVec(bml.stat.randperm_group(group{2}));
        end
        
        ests = cell(1, n_shuf);
        [ests{1}, ~, n_tr_in_fr] = EnIxn.GLM.lev_ixn( ...
            ens_mat, ...
            ch, ...
            cond, ...
            'lev_kind', lev_kind, ...
            'smooth_fr', smooth_fr);
        
        if n_shuf > 1
            parfor i_shuf = 2:n_shuf % parfor
                ests{i_shuf} = EnIxn.GLM.lev_ixn( ...
                    {ens_mat{1}(ix_shuf{i_shuf,1},:), ...
                     ens_mat{2}(ix_shuf{i_shuf,2},:)}, ...
                    ch, ...
                    cond, ...
                    'lev_kind', lev_kind, ...
                    'smooth_fr', smooth_fr); %#ok<PFBNS>
            end
        end
        
        %% Output
        % Collect ests
        ests = cat(3, ests{:});
        est = median(ests, 3);
        ci = cat(3, ...
            prctile(ests, 2.5, 3), ...
            prctile(ests, 97.5, 3));
        
        %%
        if dim_rel_W == 2
            est = est';
            ci = permute(ci, [2, 1, 3]);
            n_tr_in_fr = n_tr_in_fr';
        end

        %% Assign
        W.est0 = ests(:,:,1);
        W.ests = ests;
        W.est = est;
        W.ci = ci;
        W.n_tr_in_fr = n_tr_in_fr;
        W.n_tr_total = W.Data.n_tr;
        
        L = struct;
        L = copyprops(L, W, 'props', W.res_props);
        L = copyFields(L, W.S0_file);
        mkdir2(fileparts(file));
        save(file, '-struct', 'L');
        fprintf('Saved results to %s.mat\n', file);
    end
end
%% Plot
methods
    function plot_and_save_all(W)
        clf;
        W.imagesc;
        file = W.get_file({'plt', 'img'});
        savefigs(file);
    end
    function imagesc(W, varargin)
        
        %% Input
        est = W.est0;
        
        % DEBUG
        est = W.est0;
%         ests = W.ests;
%         ests = cat(3, ests{:});
%         ci = cat(3, ...
%             prctile(ests, 2.5, 3), ...
%             prctile(ests, 97.5, 3));        
        
        ci = W.ci;
        
        prop_tr = W.n_tr_in_fr ./ W.n_tr_total;
        thres_plot_prop = W.thres_plot_prop;
        
        %% Threshold
        img_incl = (prop_tr > thres_plot_prop);
        if W.n_shuf > 1
            img_incl = img_incl & (est < ci(:,:,1)) | (est > ci(:,:,2));
        end
        
%         img_incl = (prop_tr > thres_plot_prop);
        
        imagesc(W.t_reg, W.t_reg, est .* img_incl);
        xlabel(sprintf('t_{%s} (s)', Data.Consts.dimNames_long{W.dim_rel_W}));
        ylabel(sprintf('t_{%s} (s)', Data.Consts.dimNames_long{W.dim_irr_W}));
        
        str = W.get_title;
        title(str);
        
        for bat = {
                1, 'XDir'
                2, 'YDir'
                }'
            [dim, d] = deal(bat{:});
            
            switch W.t0_kinds{dim}
                case 'st'
                    set(gca, d, 'normal');
                case 'en'
                    set(gca, d, 'reverse');
            end
        end
        
        bml.plot.beautify;
        axis square;
        for offset = [-W.smooth_ms / 1e3, 0, +W.smooth_ms / 1e3]
            hold on;
            if offset == 0
                spec = {'w', 'LineStyle', '--', 'LineWidth', 2};
            else
                spec = {'w', 'LineStyle', ':', 'LineWidth', 2};
            end
            h = crossLine('NE', offset, spec);
            uistack(h, 'top');
        end       
        hold off;
    end
    function plot(W)
    end
end
%% Convenience
methods
    function v = get.t_st_reg_fr(W)
        v = W.Time.convert_sec2fr_ix(W.t_st_reg_ms / 1e3);
    end
    function v = get.t_en_reg_fr(W)
        v = W.Time.convert_sec2fr_ix(W.t_en_reg_ms / 1e3);
    end
    function t_reg = get.t_reg(W)
        t_reg = W.Time.convert_fr2sec(W.t_st_reg_fr:W.t_en_reg_fr);        
    end
    function t0_irr = get.t0_irr(W)
        if W.t0_opp
            t0s = {'st', 'en'};
            assert(ismember(W.t0_rel, t0s));
            t0_irr = setdiff(t0s, W.t0_rel);
            t0_irr = t0_irr{1};
        else
            t0_irr = W.t0_rel;
        end            
    end
    function v = get.t0_kinds(W)
        v = {W.t0_rel, W.t0_irr};
        if W.dim_rel_W == 2
            v = v([2, 1]);
        end
    end
end
%% File
methods
    function v = get_file_fields0(W)
        fs = W.get_file_fields0@EnIxn.GLM.MainGLM;
        fs = fs(~strcmp('t0_kind', fs(:,1)), :);
        
        v = union_general(fs, {
            ... 't_st_reg_ms', 'stl'
            ... 't_en_reg_ms', 'enl'
            'smooth_ms', 'sm'
            'n_shuf', 'nsf'
            'seed', 'sd'
            't0_rel', 't0r'
            't0_opp', 't0o'
            }, 'stable', 'rows');
    end
end
end