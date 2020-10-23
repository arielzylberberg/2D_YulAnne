classdef DtbCalcCollectionSim < Fit.D2.RT.BoundedCondEn.DtbCalcSim
    %
    % 2015 YK wrote the initial version.
properties (Access = private)
    DtbCalcs = {};
    n_rep_DtbCalc = 1; % Set to >= 2 for testing.
    ds_param_batch = dataset;
end
methods
    function W = DtbCalcSimsCollection(DtbCalcs)
        W.add_deep_copy({'DtbCalcs'});
        if nargin > 0
            W.DtbCalcs = DtbCalcs;
        end
    end
    function init(W, varargin)
        % may get p_dim1_1st and n_rep_DtbCalc, in addition to inherited props.
        varargin2props(W, varargin);
    end
end
%% Facade
methods
    function [td, ch, traj] = calc_dtb(W, varargin)
        W.init(varargin{:});
        [td, ch, traj] = W.get_pred_td_tr_t_ch;
    end    
end
%% Calculation
methods
    function [td_pdfs, trajs] = get_pred_td_tr_t_ch_pdf(W)
        % Summarize results from get_pred_td_tr_t_ch
        % per batch.
        %
        % td_pdfs{i_batch} = td_pdf
        % td_pdf : t x tr x 1 x ch1 x ch2
        
        % td: tr x 1
        % ch: tr x n_dim
        % traj: tr x dim x t
        [td, ch, traj] = W.get_pred_td_tr_t_ch;
        
        n_batch = W.get_n_DtbCalcs;
        ix_tr = W.get_ix_tr;
        ix_batch = W.get_ix_batch;
        
        td_pdfs = cell(n_batch, 1);
        trajs = cell(n_batch, 1);
        
        n_tr_all = W.get_n_tr_all;
        n_ch_all = W.get_n_ch_all;
        n_rep_all = W.get_n_rep_DtbCalc;
        ix_rep = W.get_ix_rep;
        nt = W.get_nt;
        
        % Some particles that are not absorbed by RT
        % is considered unabsorbed at all.
        td_exists = ~isnan(td) & all(ch ~= 0, 2); 
        
        for i_batch = 1:n_batch
            incl = ix_batch == i_batch;
            n_tr = n_tr_all(i_batch);
            n_ch = n_ch_all(i_batch);
            n_rep = n_rep_all(i_batch);
            
            incl = incl & td_exists;
            
            td_pdfs{i_batch} = accumarray( ...
                [td(incl), ix_tr(incl), ones(nnz(incl), 1), ch(incl, :)], ...
                1, ...
                [nt, n_tr, 1, n_ch, n_ch], ...
                @sum) ...
                / n_rep; % Average within batch across repetitions
                        
            incl_traj = incl & (ix_rep == 1);
            trajs{i_batch} = traj(incl_traj);
        end
    end
    %% Meta-set
    function add_DtbCalc(W, DtbCalc, batch_params)
        % add_DtbCalc(W, DtbCalc, batch_params)
        if iscell(DtbCalc)
            DtbCalc = Fit.D2.RT.BoundedCondEn.DtbCalcSim(DtbCalc{:});
        end
        W.DtbCalcs{end+1} = DtbCalc;
        n_DtbCalcs = W.get_n_DtbCalcs;
        
        S = varargin2S(batch_params);
        W.set_ds_param_batch(ds_set(W.get_ds_param_batch, ...
            n_DtbCalcs + 1, S));
    end
    function remove_DtbCalc_all(W)
        W.DtbCalcs = {};
        W.ds_param_batch = dataset;
    end
    function set_ds_param_batch(W, v)
        W.ds_param_batch = v;
    end
    function v = get_ds_param_batch(W)
        v = W.ds_param_batch;
    end

    %% Individual properties - flags
    function v = get_ix_batch(W)
        v = W.get_all('ix_batch');
    end
    function v = get_ix_tr(W)
        % original trial number
        v = W.get_all('ix_tr');
    end
    function v = get_ix_rep(W)
        % repetition number within each batch
        v = W.get_all('ix_rep');
    end
    
    %% Meta-get
    function v = get_all(W, prop, varargin)
        S = varargin2S(varargin, {
            'repeat_call', false
            });
        
        n = W.get_n_DtbCalcs;
        v = cell(n, 1);
        n_rep = W.get_n_rep_DtbCalc;
        DtbCalcs = W.get_DtbCalcs;                    
        
        for ii = 1:W.get_n_DtbCalcs
            DtbCalc = DtbCalcs{ii};
            
            switch prop
                case {'ix_batch', 'ix_tr', 'ix_rep'}
                    n_tr = DtbCalc.get_n_tr;
                    switch prop
                        case 'ix_batch'
                            % Calculated here in accordance with other properties
                            v{ii} = zeros(n_tr * n_rep(ii), 1) + ii;
                        case 'ix_tr'
                            % Calculated here in accordance with other properties
                            v{ii} = repmat((1:n_tr)', [n_rep(ii), 1]);
                        case 'ix_rep'
                            % Calculated here in accordance with other properties
                            v{ii} = vVec(repmat(1:n_rep, [n_tr, 1]));
                    end
                    assert(size(v{ii}, 1) == n_tr * n_rep(ii));
                    
                otherwise
                    if S.repeat_call
                        n_rep_call = n_rep(ii);
                        vv = cell(n_rep_call, 1);
                        for jj = 1:n_rep_call
                            vv{jj} = DtbCalc.(['get_' prop]);
                        end
                        v{ii} = cat(1, vv{:});
                    else
                        v{ii} = repmat(DtbCalc.(['get_' prop]), [n_rep(ii), 1]);
                    end
            end
        end
        v = cat(1, v{:});
    end
    function v = get_n_DtbCalcs(W)
        v = numel(W.DtbCalcs);
    end
    function v = get_n_rep_DtbCalc(W)
        n_DtbCalcs = W.get_n_DtbCalcs;
        v = rep2fit_strict(W.n_rep_DtbCalc, [n_DtbCalcs, 1]);
    end
    
    %% Individual properties - private, non-expanded
    function set_DtbCalcs(W, v)
        W.DtbCalcs = v;
    end
    function v = get_DtbCalcs(W)
        v = W.DtbCalcs;
    end

    function set_n_rep_DtbCalc(W, v)
        W.n_rep_DtbCalc = v;
    end

    function set_p_dim1_1st(W, v)
        W.p_dim1_1st = v;
    end
    function v = get_p_dim1_1st(W)
        v = W.p_dim1_1st;
    end
    
    %% Individual properties - delegated, non-expanded
    function v = get_n_tr(W)
        v = sum(W.get_n_tr_all);
    end
    function v = get_n_tr_all(W)
        v = W.get_all('n_tr');
    end
    
    function v = get_n_ch(W)
        v = W.get_all('n_ch');
        v = v(1,:); % Use ther first row
    end
    function v = get_n_ch_all(W)
        v = W.get_all('n_ch');
    end
    
    function v = get_n_dim(W)
        v = W.get_n_dim_all;
        v = v(1,:);
    end
    function v = get_n_dim_all(W)
        v = W.get_all('n_dim');
    end
    
    %% Individual properties - delegated, expanded
    function v = get_drift(W)
        v = W.get_all('drift');
    end
    function v = get_bound(W)
        v = W.get_all('bound');
    end
    function v = get_tnd_st(W)
        % tnd_st should be always different for each realization
        v = W.get_all('tnd_st', 'repeat_call', true); % Special
    end
    function v = get_sigmaSq_fac_bef_start(W)
        v = W.get_all('sigmaSq_fac_bef_start');
    end
    function v = get_sigmaSq_fac_together(W)
        v = W.get_all('sigmaSq_fac_together');
    end
    function v = get_drift_fac_together(W)
        v = W.get_all('drift_fac_together');
    end
    function v = get_use_gpu(W)
        v = W.get_all('use_gpu');
    end
end
end