% Compute p_acq given a

%% Init
clear;
init_path;

%% Load BoundedEn fits
file_fits = '../Data_2D/IxnKernel.main_ixn_kernel/main_lev_pred_vs_data/sbj={DX,MA,VL}+prd=RT+tsk={H,V}+dif=[1,2].mat';
L_fits = load(file_fits);
Ws = L_fits.Ws;

% %% Test
% log_p_ch = get_log_p_ch_given_p_acq( ...
%     p_acq, ev_itv1, ch_data, params_ch_a, d_cond);
% disp(exp(log_p_ch));

%% Prepare loop
subjs = Data.Consts.subjs_RT;

n_dim = 2;
n_subj = numel(subjs);

dim_incl = 1:n_dim;
subj_incl = 1:n_subj;

global to_use_parallel
to_use_parallel = false;

n_itv = 2;
dif_incl = 1; % 1:2;
truncate_st_fr = 15;
n_fr_in_itv = 20;

ress = cell(n_subj, n_dim);
pth = '../Data_2D/IxnKernel.PAcqD1';
mkdir2(pth);

%% Loop
for dim_rel = dim_incl
    for i_subj = subj_incl
        %% Prepare W
        subj = subjs{i_subj};
        
        W = Ws{i_subj, dim_rel};
        W.pred;
        
        %% Prepare for saving
        S_file = varargin2S({
            'sbj', subj
            'prd', W.parad
            'dtk', W.n_dim_task
            'dmr', dim_rel
            'dif', dif_incl
            'trst', truncate_st_fr
            'friv', n_fr_in_itv
            'nitv', n_itv
            });

        %% Get variables from W
        ch = W.Data.ch;
        cond = W.Data.cond;
        ev = W.Data.en_rel_mat;
        RT_pred_pdf_tr = W.Data.RT_pred_pdf_tr;
        
        info_dat = packStruct(ch, cond, ev, RT_pred_pdf_tr);

        %% Preprocess data
        n_tr = size(ch, 1);
        trs = (1:n_tr)';
        ch_data = accumarray([trs, ch], 1) == 1; % Make it logical

        % ch_pred(tr, [ch0, ch1])
        ch_pred = permute(sum(RT_pred_pdf_tr, 1), [2, 3, 1]);
        ch_pred = ch_pred ./ sum(ch_pred, 2);
        
        %% Run analysis
        [p_acq_given_s, info] = IxnKernel.PAcqD1.get_p_acq( ...
            ev, ch_data, ch_pred, cond, ...
            'dif_incl', dif_incl, ...
            'truncate_st_fr', truncate_st_fr, ...
            'n_fr_in_itv', n_fr_in_itv, ...
            'n_itv', n_itv);

        info.info_dat = info_dat;
        ress{i_subj, dim_rel} = info;
        
        %% Plot p_acq_given_a_s
        p = info.p_acq_given_a_s;
        p(p == 0) = nan;
        
        clf;
        imagesc(info.s, info.a, p);
        xlabel('s');
        ylabel('a');
        axis square;
        set(gca, 'XTick', info.s, 'YTick', info.a);
        bml.plot.beautify;
%         bml.plot.colormap_nan;
        
        eprintf info.p_acq_given_a_s

        S_file1 = varargin2S({
            'plt',' p_acq_given_a_s'
            }, S_file);
        nam = bml.str.Serializer.convert(S_file1);
        
        title(bml.str.get_title(nam));
        savefigs(fullfile(pth, nam));
        
        %% Plot p_a_given_s
        p = info.p_a_given_s;
        p(p == 0) = nan;
        
        clf;
        imagesc(info.s, info.a, p);
        xlabel('s');
        ylabel('a');
        axis square;
        set(gca, 'XTick', info.s, 'YTick', info.a);
        bml.plot.beautify;
%         bml.plot.colormap_nan;
        
        eprintf info.p_a_given_s

        S_file1 = varargin2S({
            'plt',' p_a_given_s'
            }, S_file);
        nam = bml.str.Serializer.convert(S_file1);
        
        title(bml.str.get_title(nam));
        savefigs(fullfile(pth, nam));
        
        %% Plot p_acq_s
        s = info.s;
        p = [info.p_acq_given_s, ...
             info.p_acq_given_s(end)];
        s1 = [0.5, s + 0.5];
        
        stairs(s1, p, 'k-');
        xlabel('s');
        ylabel('p_{acq|s}');
        ylim([0, 1])
        set(gca, 'XTick', s, 'YTick', 0:0.2:1);
        bml.plot.beautify;
        
        S_file1 = varargin2S({
            'plt',' p_acq_given_s'
            }, S_file);
        nam = bml.str.Serializer.convert(S_file1);
        
        title(bml.str.get_title(nam));
        savefigs(fullfile(pth, nam));
        
    end
end

%% Save mat
S_file = varargin2S({
    'sbj', subjs
    'dmr', dim_incl
    }, S_file);
nam = bml.str.Serializer.convert(S_file);
file = fullfile(pth, nam);
save([file, '.mat'], 'ress', 'S_file', 'L_fits', 'file_fits');
fprintf('Saved to %s.mat\n', file);

%% Imgather


% %% Test
% f_p_ch_given_util = IxnKernel.PAcqAD1.get_f_p_ch_given_util( ...
%     ev_itv, ch_pred, ch_data);
% 
% p_util = ...
%     IxnKernel.PAcqAD1.get_p_util( ...
%         ch, ev_itv, f_p_ch_given_util);
% 
% %%
% EP = Rev.EP;

%%
