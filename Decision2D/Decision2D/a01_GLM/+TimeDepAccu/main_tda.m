%%
init_path;

% GLM.ModelFreeChRt.get_pred_pdf_all_inh; % Run only once

%%
% file = '../Data_2D/Fit.ModelFreeChRt/get_pred_pdf_all_inh';
% L_data_pred = load(file);
% datas = L_data_pred.datas;
% preds = L_data_pred.preds;

file = '../Data_2D/Fit.RewardRate.main_reward_rate/pred_data_by_model';
L_data_pred = load(file);
datas = L_data_pred.data;
preds = L_data_pred.pred;
models = L_data_pred.models;

parad = 'RT';
fprintf('Loaded data & prediction from %s\n', file);

%%
pth = '../Data_2D/TimeDepAccu.main_tda';
subjs = Data.Consts.subjs_RT;
n_subj = numel(subjs);
n_dim = 2;

for model1 = {'min_sub', 'mdl_Ser', 'min_sup', 'mdl_Par'}
    model = model1{1};
    clf;

    ax = gobjects(n_subj, n_dim);
    for i_subj = 1:n_subj
        for dim = 1:n_dim
            subj = subjs{i_subj};
            ax(i_subj, dim) = subplotRC(n_subj, n_dim, i_subj, dim);

            i_model = find(strcmp(models, model));

            for source1 = {'data', 'pred'}
                source = source1{1};

                switch source
                    case 'pred'
                        p_rt = preds{i_subj,i_model};
                        p_rt = bsxfun(@times, ...
                            p_rt, sums(datas{i_subj}, [1, 4, 5]));

                        linewidth = 2;
                    case 'data'
                        p_rt = datas{i_subj};
                        linewidth = 0.5;
                end

                TimeDepAccu.plot_tda(p_rt, 'dim', dim, ...
                    'to_plot_shade', strcmp(source, 'data'), ...
                    'linewidth', linewidth);
                hold on;
            end
            hold off;
        end
        sameAxes(ax(i_subj,:), [], [], 'x');
    end
    savefigs(fullfile(pth, ...
        sprintf('prd=%s+mdl=%s+plt=tda', model, parad)));
end