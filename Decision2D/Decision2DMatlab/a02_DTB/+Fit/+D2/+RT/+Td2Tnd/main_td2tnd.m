% function main_td2tnd()

% Compare Ser vs Par in a model-free way
clear;
init_path;

%% == Common settings
C0 = varargin2C({
    'to_use_easiest_only', {0, 1}
    'to_exclude_bins_wo_trials', 10 % 10 or []
    ...
    'td_kind', {'Ser', 'Par'}
    'skip_existing_mat', true
    'to_plot', false  % CHECKING - plotting takes time but nice for sanity check
    % REMOVE begin
    'ub_mu_margin', 2 % 2 or []
    'ub_sd_margin', 2 % 2 or []
    'max_sd_mu_ratio', nan % 2, ... % [] or 1.5
    'fix_miss', false
    ...
    ...
    'dif_rel_incl', 1:6  % Option to exclude easiest: 1=hardest; 6:6-th hardest; 1:6 includes all difficulty levels
    'dif_irr_incl', 1:6  % Option to exclude easiest
    % REMOVE end
    });
S0 = varargin2S(C0);

max_iter = 500;
% max_iter = 0;  % CHECKED

% To run on a subset, use, e.g., Data.Consts.subj_parad_td2tnd(1:3, :)
subj_parads = Data.Consts.subj_parad_td2tnd;
subj_parads_par = row2cell(subj_parads);

parads = {'RT', 'unimanual', 'bimanual'};

% REMOVE begin
% i_bat_incl = [1, 4, 19];
% REMOVE end
i_bat_incl = 1:length(subj_parads_par);

%% == Fit to real data
for i_bat = i_bat_incl
    bat = subj_parads_par{i_bat};
    [subj, parad] = deal(bat{:});
    
    for to_use_easiest_only = cell2mat(S0.to_use_easiest_only)
        W0 = Fit.D2.RT.Td2Tnd.Main;
        C = varargin2C({
                'subj', {subj}
                'parad', {parad}
                'MaxIter', max_iter  % CHECKING  % README: 0 to check without fitting; 1e4 for real fitting - never actually goes that much though
                'UseParallel', 'never'
                'to_use_easiest_only', to_use_easiest_only
                'to_use_easiest_only_for_fit', to_use_easiest_only
                'to_use_easiest_only_for_comparison', -to_use_easiest_only
            }, C0);
        % To see the code, type:
        % edit Fit.D2.RT.Td2Tnd.Main.batch
        W0.batch(C{:});
    end
end

%% == Batch comparison - real data
for parad = parads 
    subjs = Data.Consts.get_subjs_parad(parad{1});
    for to_use_easiest_only = cell2mat(S0.to_use_easiest_only)
        W0 = Fit.D2.RT.Td2Tnd.Main;
        W0.init;

        W0.to_use_easiest_only_for_fit = to_use_easiest_only;
        W0.to_use_easiest_only_for_comparison = -to_use_easiest_only;
        W0.fix_miss = S0.fix_miss;
        W0.ub_mu_margin = S0.ub_mu_margin;
        W0.ub_sd_margin = S0.ub_sd_margin;
        W0.max_sd_mu_ratio = S0.max_sd_mu_ratio;
        W0.to_exclude_bins_wo_trials = S0.to_exclude_bins_wo_trials;

        W0.compare_models( ...
            'subj', subjs, ...
            'parad', parad{1}, ...
            'force_to_use_easiest_only_for_comparison', ...
                -to_use_easiest_only);
    end
end

%% == Batch plot
% for parad = {'RT'}
%     for difs = {
%             0, 1
%             }
%         W0 = Fit.D2.RT.Td2Tnd.Main;
% %         W0.to_use_easiest_only_for_fit = difs{1};
% %         W0.to_use_easiest_only_for_comparison = -1;
% 
%         ec = -1;
% %         if difs{1} == 0
% %             ec = 0;
% %         else
% %             ec = -difs{1};
% %         end
% 
%         W0.batch_plot( ...
%             'parad', parad{1}, ...
%             'to_use_easiest_only_for_fit', difs{1}, ...
%             'ub_mu_margin', 2, ... % 2, ... % 2 or []
%             'ub_sd_margin', 2, ... % 2, ... % 2 or []
%             'max_sd_mu_ratio', nan, ... % nan, ... % nan, ... % [] or 1.5
%             'fix_miss', {S0.fix_miss}, ... 
%             'to_exclude_bins_wo_trials', 10, ...
%             'to_use_easiest_only_for_comparison', ec);
%     end
% end

%% == Simulate data
for i_bat = i_bat_incl
    bat = subj_parads_par{i_bat};
    [subj, parad] = deal(bat{:});

    for to_use_easiest_only = cell2mat(S0.to_use_easiest_only)
        for td_kind = {'Ser', 'Par'}
            C = varargin2C({        
                'subj', subj
                'parad', parad
                'td_kind', td_kind{1}
                'to_use_easiest_only', ...
                    to_use_easiest_only
                }, C0);
            Fit.D2.RT.Td2Tnd.simulate_td2tnd(C{:});
        end
    end
end

%% == Fit simulated data
W0 = Fit.D2.RT.Td2Tnd.Main;
td_sims = {'Ser', 'Par'};
for i_bat = i_bat_incl
    bat = subj_parads_par{i_bat};
    [subj0, parad] = deal(bat{:});
    
    for i_td_sim = 1:length(td_sims) % the model that simulated
        for to_use_easiest_only = cell2mat(S0.to_use_easiest_only)
            subj = Fit.D2.RT.Td2Tnd.get_sim_subj( ...
                subj0, td_sims{i_td_sim}, 1, to_use_easiest_only);
        
            C = varargin2C({
                'parad', parad
                'to_use_easiest_only', to_use_easiest_only
                'to_use_easiest_only_for_fit', to_use_easiest_only
                'to_use_easiest_only_for_comparison', -to_use_easiest_only
                'td_kind', {'Ser', 'Par'}  % the model that fits
                'subj', subj
                ...
                'MaxIter', max_iter  % 0 to check without fitting; 1e4 for real fitting
                'UseParallel', 'never'
                }, C0);
            W0.batch(C{:});
        end
    end
end

%% == Batch comparison - simulated data
for parad = parads
    for to_use_easiest_only = cell2mat(S0.to_use_easiest_only)
        subjs0 = Data.Consts.get_subjs_parad(parad{1});
        n_subj = 0;
        subjs = {};            
        for td = {'Ser', 'Par'}
            for subj = subjs0(:)'
                for ef = to_use_easiest_only
                    n_subj = n_subj + 1;
                    subjs{n_subj} = Fit.D2.RT.Td2Tnd.get_sim_subj( ...
                        subj{1}, td{1}, 1, ef); %#ok<SAGROW>
                end
            end
        end
        
        W0 = Fit.D2.RT.Td2Tnd.Main;
        W0.init;

        W0.to_use_easiest_only_for_fit = to_use_easiest_only;
        W0.to_use_easiest_only_for_comparison = -to_use_easiest_only;
        W0.fix_miss = S0.fix_miss;
        W0.ub_mu_margin = S0.ub_mu_margin;
        W0.ub_sd_margin = S0.ub_sd_margin;
        W0.max_sd_mu_ratio = S0.max_sd_mu_ratio;
        W0.to_exclude_bins_wo_trials = S0.to_exclude_bins_wo_trials;
        W0.compare_models( ...
            'subj', subjs, ... 
            'parad', parad{1}, ...
            ...
            'force_to_use_easiest_only_for_comparison', ...
                -to_use_easiest_only);
    end
end

