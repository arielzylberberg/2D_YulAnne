%% Load AnneVD
clear;
init_path;

file_in = '../Data_2D/sTr/AnneVD.mat';
tbl = struct2table(load(file_in));
fprintf('Loaded %s\n', file_in);
n_tr = size(tbl, 1);

%% Load RDK
file_in0 = '../Data_2D/orig_Anne_VD_eye/dataAll_ID2_2D.mat';
L0 = load(file_in0);
fprintf('Loaded %s\n', file_in0);

RDM = L0.dsgn.RDM;
RDK = PsyRDKConst;

pth_out = '../Data_2D/an00_import/main_filter_ME_AnneVD';

%% Construct MFilt
dur_sec = 5;
n_pix = ceil(RDM.ppd * 2.5) * 2 + 1;

C = varargin2C({
    'degPerPix', 1 / RDM.ppd
    'dt', 1 / RDM.fps
    'sigX', 0.35 % same as Kiani 2008
    'apRPix', RDM.ppd * 2.5
    'lenT', RDM.fps * dur_sec
    'k', RDM.fps
    });
MFilt = ConvMotionFilter([], C{:});

%% Plot filters
for filt_kind = {'tx', 'ty', 'xy'}
    fig_tag(['motion_filter_', filt_kind{1}]);
    clf;
    MFilt.plot_filt(filt_kind{1});
    for ii = 1:4
        subplot(2,2,ii); 
        switch filt_kind{1}
            case {'tx', 'ty'}
                xlim([0, 0.3]); 
                ylim([-2.5, 2.5]); 
        end
        if ii ~= 3
            set(gca, 'XTickLabel', '', 'YTickLabel', '');
            xlabel('');
            ylabel('');
        end
    end
    savefigs(fullfile(pth_out, ...
        sprintf('filter_motion_%s', filt_kind{1})), ...
        'size', [400, 400]);
end

%% Plot imulse response

% % Flipping bwPix itself - deprecated
% xyct = MFilt.get_xyct_impulse();
% bwPix1 = MFilt.xyct2full(xyct);
% bwPix2 = flip(bwPix1, 2);
% en1 = MFilt.apply(bwPix1, 'input', 'bwPix');
% en2 = MFilt.apply(bwPix2, 'input', 'bwPix');

fig_tag('impulse');
clf;
[h, ME_pos] = MFilt.plot_impulse;
set(h, 'Color', 'b');
hold on;
h_right_positive = h;


[h, ME, t] = MFilt.plot_impulse('direction', -1);
set(h, 'Color', 'r');
hold on;
h_left_positive = h;

h_right_negative = plot(t, -ME_pos, 'bo');
hold on;

h_left_negative = plot(t, -ME, 'ro');
hold off;

crossLine('h', 0, 'k:');

legend([h_right_positive, h_left_positive, ...
        h_right_negative, h_left_negative], ...
    {'+Right', '+Left', '-Right', '-Left'});

xlabel('time (s)');
ylabel('motion energy (a.u.)');

csv_file = fullfile( ...
    '../Data_2D/sTr/', ...
    sprintf('impulse_MotionEnergy_refreshrate=%d.csv', ...
        round(1 / MFilt.dt)));
csvwrite(csv_file, ME_pos(:));
fprintf('Exported kernel to %s\n', csv_file);

savefigs(fullfile(pth_out, 'impulse'));

%% Choose trials
seed = cellfun(@xyct2seed_id, tbl.xyct);
[~,~,ix_seed] = unique(seed);

% tr_incl = find(ix_seed <= 10);
tr_incl = 1:n_tr; % 1:100; % 
% tr_incl = 1:100; % :500;

n_tr_incl = length(tr_incl);
tbl1 = tbl(tr_incl, :);

%% Check xyct vs bwPix
fig_tag('xyct vs bwPix');
clf;
xyct = tbl1.xyct{2};
MFilt.plot_xyct(xyct, 'fr', 5);
% savefigs(fullfile(pth_out, 'xyct_vs_bwPix'));

%% Compute motion energy
if ~ismember('en', tbl.Properties.VariableNames) ...
        || inputYN('tbl.en exists! Recompute (y/n)? ')
    %%
    xyct = tbl1.xyct;
    t_st = tic;
    en = cell(n_tr_incl, 2);

    for tr = 1:n_tr_incl
        en{tr, 1} = MFilt.apply(xyct{tr}); %#ok<PFBNS>
    end
    t_el = toc(t_st);
    fprintf('Processing %d trials took %1.1g seconds\n', ...
        n_tr_incl, t_el);
    tbl1.en(:, 1) = en(:, 1);
    
    %%
    if ~ismember('en', tbl.Properties.VariableNames)
        tbl.en = cell(n_tr, 2);
        fprintf('Created column en in tbl\n');
    end
    tbl.en(tr_incl, 1) = en(:,1);
    fprintf('Updated tbl.en\n');
end

%% Get impulse response
[impulse_ME, t] = MFilt.get_impulse_response;
impulse_ME = impulse_ME(:)';
impulse_ME = impulse_ME / sum(impulse_ME);

%% Convolve rCE with impulse_ME to get CE
% Just to equate everything
tr_incl = 1:n_tr;
n_tr_incl = numel(tr_incl);
en_color = cell(n_tr_incl, 1);
dim_color = 2;
raw_en = tbl.raw_en(tr_incl, dim_color);
z = zeros(1, length(impulse_ME) - 1);
for i_tr = 1:n_tr_incl
    en_color{i_tr} = conv_t([raw_en{i_tr}, z], impulse_ME);
end
tbl.en(tr_incl(:), dim_color) = en_color;

%% Example color energy
color_coh = tbl1.condC(tr_incl);
tr_pos = find(color_coh > 0);
tr_pos = tr_pos(end);
tr_neg = find(color_coh < 0);
tr_neg = tr_neg(1);

fig_tag('example');
clf;
colors = {'b', 'r'};
trs = [tr_pos, tr_neg];
for i_tr = 2:-1:1
    hs(i_tr) = subplot(2,1,i_tr);
    tr = trs(i_tr);
    plot(raw_en{tr}, 'o:', 'color', colors{i_tr});
    hold on;
    plot(en_color{tr}, '-', 'color', colors{i_tr});
    hold on;
    title(sprintf('Trial %d, coh %1.3f', tr, color_coh(tr)));
end
sameAxes(hs);

%% Check symmetry
% When checking, exclude trials with tr_excl(trial) == 1 
% (first trials for each subject ID=2,3 where the color convention is
% different from the rest of the trials
tbl1 = tbl(tr_incl, :);
tbl1 = tbl1(~tbl1.to_excl, :);
n_tr_incl = size(tbl1, 1);

for dim = 1:2
    seed = cellfun(@xyct2seed_id, tbl1.xyct);
    [~,~,ix_seed] = unique(seed);
    ix_sign_coh = sign(tbl1.cond(:,dim)) * 0.5 + 1.5;
    tr_plot = accumarray([ix_seed, ix_sign_coh], (1:n_tr_incl)', [], @sum);
    n_seed = max(ix_seed);

    fig_tag('Symmetry');
    clf;
    en = tbl1.en(:,dim);

    max_dif = -inf;
    max_tr = [nan, nan];

    max_dif_prop = -inf;
    max_dif_prop_tr = [nan, nan];

    for ii = 1:n_seed
        tr_plot1 = tr_plot(ii, :);
        if all(tr_plot1)
            plot(en{tr_plot1(1)}, en{tr_plot1(2)}, 'ko');

            max_dif1 = max(abs(en{tr_plot1(1)} + en{tr_plot1(2)}));
            if max_dif1 > max_dif
                max_dif = max_dif1;
                max_tr = tr_plot1;
            end

            dif_prop = max_dif1 / max(abs(en{tr_plot1(1)}));
            if dif_prop > max_dif_prop
                max_dif_prop = dif_prop;
                max_dif_prop_tr = tr_plot1;
            end

            hold on;
        end
    end

    crossLine('SE', 0, 'k:');
    axis equal
    axis square

    xlabel('en (negative coh)');
    ylabel('en (positive coh)');

    try
        h = findobj('type', 'text');
        delete(h);
    catch
    end
    bml.plot.text_align({
        sprintf('Max_{trial} abs(difference): %1.2g', max_dif)
        sprintf('Max_{trial}( abs(dif)/abs(max) ): %1.2g', max_dif_prop)
        }, ...
        'corner', 'NE');
    fprintf('Max abs difference - trials: %d\n', max_tr);
    fprintf('Max abs difference proportion - trials: %d\n', max_dif_prop_tr);

    savefigs(fullfile(pth_out, ...
        sprintf('plt=symmetry+dim=%d', dim)), 'ext', {'png', 'eps'});
end

%% Plot cond vs en
for dim = 1:2
    m_en = cellfun(@nanmean, tbl1.en(:,dim));
    m_r_en = cellfun(@nanmean, tbl1.raw_en(:,dim));

    cond = tbl1.cond(:,dim);
    [conds, ~, dcond] = unique(cond);
    m_r_en_cond = accumarray(dcond, m_r_en, [], @nanmean);
    m_en_cond = accumarray(dcond, m_en, [], @nanmean);
    e_en_cond = accumarray(dcond, m_en, [], @nanstd);

    fig_tag('cond vs raw en');
    clf;
    plot(conds, m_r_en_cond, 'ko');
    crossLine('h', 0, 'k--');
    crossLine('v', 0, 'k--');
    xlabel('coherence');
    ylabel('raw en');
    savefigs(fullfile(pth_out, ...
        sprintf('plt=cond_vs_raw_en+dim=%d', dim)));

    fig_tag('cond vs en)');
    clf;
    plot(cond, m_en, 'ko')
    crossLine('h', 0, 'k--');
    crossLine('v', 0, 'k--');
    xlabel('coherence');
    ylabel('en (mean wi trial)');
    savefigs(fullfile(pth_out, ...
        sprintf('plt=coh_vs_en_wi_trial+dim=%d', dim)));

    fig_tag('cond vs mean en');
    clf;
    errorbar(conds, m_en_cond, e_en_cond, 'ko');
    crossLine('h', 0, 'k--');
    crossLine('v', 0, 'k--');
    xlabel('coherence');
    ylabel('en (mean wi coh +- SD)');
    savefigs(fullfile(pth_out, ...
    sprintf('plt=coh_vs_en_wi_coh+dim=%d', dim)));
end

%% Save
S_tbl = table2struct(tbl, 'ToScalar', true);
save(file_in, '-struct', 'S_tbl');
fprintf('Saved en to %s\n', file_in);
