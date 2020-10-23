function subj_name = simulate_dtb(W, varargin)

%% Get RT_pred_pdf from Ser and Par
S = varargin2S(varargin, {
    'seed', 1
    'desc', ''
    'src', 'pred' % 'pred'|'data'
    'subj', ''
    'get_name_only', false
    });

if S.get_name_only
    if isempty(S.subj)
        S.subj = W.Data.subj;
    end
    subj_name = sprintf('%s_%s_seed%d', ...
        S.subj, S.desc, S.seed);
    return;
end

%% Load fit
file = [W.get_file, '.mat'];
assert(exist(file, 'file') == 2, 'File not found: %s\n', file);

L = load(file);
fprintf('Loaded fit from %s\n', file);

Fl = L.Fl;
Fl.res2W;
W = Fl.W;

switch S.src
    case 'pred'
        RT_src_pdf = W.Data.RT_pred_pdf;
    case 'data'
        RT_src_pdf = W.Data.RT_data_pdf;
end

dCond = W.Data.get_dCond;
t = W.t;

%% Simulate data from RT_pred_pdf by sampling
[~, ch_new, rt_vec_new] = Fit.simulate_data_given_pred( ...
    RT_src_pdf, dCond, t, 'seed', S.seed);

%% Save
ds00 = W.Data.ds0;
W.Data.ch = ch_new;
W.Data.rt = rt_vec_new;

L = struct;
L.dat = W.Data.ds0;
L.dat = L.dat(:, {
    'task', 'succT', 'i_all_Run', 'i_all_Tr', 'i_Tr', ...
    'subjM', 'subjC', 'condM', 'condC', 'corrM', 'corrC', 'RT'
    }); %#ok<STRNU>

subj_name = sprintf('%s_%s_seed%d', ...
    W.Data.subj, S.desc, S.seed);
subj0 = W.Data.subj;
file0 = W.Data.get_path;
W.Data.path = '';
W.Data.subj = subj_name;
file_new = W.Data.get_path;
if exist(file_new, 'file')
    error('File already exists: %s\n', file_new);
end
W.Data.subj = subj0;
W.Data.path = file0;

save(file_new, '-struct', 'L');
fprintf('Saved to %s\n', file_new);
W.Data.ds0 = ds00;
