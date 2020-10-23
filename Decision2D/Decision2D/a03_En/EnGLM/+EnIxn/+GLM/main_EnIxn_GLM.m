%% Run across tasks
init_path;
W0 = EnIxn.GLM.MainGLM;
W0.batch( ...
    'lev_kind', 'useixn', ... 'cumpred', ...
    'subj', {'VL'}, ...
    'n_dim_task', {2}, ... % {1,2}
    'dim_rel_W', {1, 2}); % {1,2}

%%
init_path;
W0 = EnIxn.GLM.MainGLM;
for lev_kind = {
        'beta'
        ... 'logit_for_ch'
        ... 'raw'
        ... 'cumpred'
        ... 'useixn'
        }'
    for n_dim_task = 1:2 % 1:2
        for dim_rel_W = 1:2 % 1:2
            W0.imgather_overlay( ...
                'lev_kind', lev_kind{1}, ...
                'n_dim_task', n_dim_task, ...
                'dim_rel_W', dim_rel_W, ...
                'truncate_first_msec', 0);
        end
    end
end

%% useixn - Compare between dim_rel_W
init_path;
W0 = EnIxn.GLM.MainGLM;

for lev_kind = {
        'useixn'
        }'
    for n_dim_task = 2
        for dif_incl = {1, 1:2}
            W0.imgather_overlay( ...
                'lev_kind', lev_kind{1}, ...
                'n_dim_task', n_dim_task, ...
                'truncate_first_msec', 0, ...
                'dif_rel_incl', dif_incl{1}, ...
                'dif_irr_incl', dif_incl{1});
        end
    end
end

%% Test 
% init_path;
% S = varargin2S({
%     'subj', 'MA'
%     'lev_cum', true
%     'lev_kind', 'useixn' % 'raw' % 'logit_for_ch'
%     'task', 'A' % 'H'
%     'dif_rel_incl', 1:2
%     'dif_irr_incl', 1:2
%     't0_kind', 'st'
%     'truncate_first_msec', 0
%     'skip_existing_mat', false
%     });
% C = S2C(S);
% 
% W = EnIxn.GLM.MainGLM(C{:});
% 
% %%
% % fig_tag('lev');
% fig_tag('cum');
% W.calculate;
% W.plot;