clear;
init_path;

%%
subjs = {'ID2', 'ID3'};
pth_in = '../Data_2D/orig_Anne_VD_eye';
file_out = '../Data_2D/sTr/AnneVD.mat';
n_dim = 2;

%%
n_subj = numel(subjs);
Ls = cell(n_subj, n_dim);
for i_subj = 1:n_subj
    %%
    subj = subjs{i_subj};
    
    for n_dim1 = 1:n_dim
        file_in = fullfile(pth_in, ...
            sprintf('dataAll_%s_%dD.mat', subj, n_dim1));
        Ls{i_subj, n_dim1} = load(file_in);
        fprintf('Loaded %s\n', file_in);
    end
end

%% Note on dataAll.MotionStimulus:
% dataAll.MotionStimulus(tr,:) is
% {x_pix, y_pix, coherent_motion, color, time_sec}
% color(frame, dot): color of the dot (0=yellow, 1=blue)
% All dots are visible on every frame

%% color condition
% cond = dat.signColorCoherence;
% [conds, ~, ix_cond] = unique(cond);
% mean_en_tr = cellfun(@(v) nanmean(v(:)), dat.MotionStimulus(:, 4));
% mean_en_cond = accumarray(ix_cond, mean_en_tr, [], @nanmean);

%% Range of color values
% color_all = dat.MotionStimulus(:,4);
% min_color = cellfun(@(v) nanmin(v(:)), color_all);
% max_color = cellfun(@(v) nanmax(v(:)), color_all);
% 
% find(min_color == 0)
% find(max_color == 1)

%%
tbl = table;
for i_subj = 1:n_subj
    for n_dim1 = 1:n_dim
        %%
        tbl1 = table;
        L = Ls{i_subj, n_dim1};
        dat = L.dataAll;
        n_tr = L.trialsTot;
        
        %% Get LLRNcol2 in preparation for roughEnCol
        RDK = PsyRDKConst;
        nDot = 4;
        % n_prop = length(L.cond.ColorCoherence) * 2;
        propRep = sort([1 - L.cond.ColorCoherence, L.cond.ColorCoherence]);
        considerAp = false;
        LLRNcol2 = RDK.LLRCol(nDot, propRep, considerAp);

        % Since conditions are symmetric, make LLRcol2 exactly symmetric.
        warning('Assuming symmetric color coherence');
        LLRNcol2 = (LLRNcol2 - flip(LLRNcol2)) / 2;

        %%
        tbl1.subj = repmat({sprintf('ID%d', L.subjID)}, [n_tr, 1]);
        tbl1.parad = repmat({'VD'}, [n_tr, 1]);
        
        tbl1.condM = dat.signMotionCoherence;
%         tbl1.condM = sign((dat.MotionDirection == 0) - 0.5) ...
%                     .* dat.MotionCoherence;
        tbl1.condC = dat.signColorCoherence;
%         tbl1.condC = dat.ColorCoherence;
%         tbl1.condC(dat.Color == 0) = 1 - tbl1.condC(dat.Color == 0);
%         tbl1.condC = logit(tbl1.condC);
%         tbl1.condC = sign(dat.Color - 0.5) ...
%                     .* dat.ColorCoherence;
        tbl1.cond = [tbl1.condM, tbl1.condC];
        
        %%
        tbl1.accuM = dat.AccuracyMotion;
        tbl1.accuC = dat.AccuracyColor;
        
        tbl1.subjM = dat.ChoiceMotion == 0;
        tbl1.subjC = dat.ChoiceColor == 1;
        tbl1.ch = [tbl1.subjM, tbl1.subjC];
        
        tbl1.task = strrep_cell(dat.Task, {
            '1D_motionVD',  'H'
            '1D_colorVD',   'V'
            '2DVD',         'A'
            });
        tbl1.dim_rel = [~strcmp(tbl1.task, 'V'), ~strcmp(tbl1.task, 'H')];
        tbl1.n_dim_task = strcmp(tbl1.task, 'A') + 1;
        
        tbl1.t_RDK_dur = dat.StimulusDuration;
        tbl1.RT = dat.RT;
        
        tbl1.i_all_Run = dat.Session;
        
        rME = cell(n_tr, 1);
        rCE = cell(n_tr, 1);
        
        en_color_mean = cell(n_tr, 1);
        en_color_LLR = cell(n_tr, 1);
        
        xyct = cell(n_tr, 1);
        coherent_motion = cell(n_tr, 1);
        
        pix_per_deg = L.dsgn.RDM.ppd;
        
        % exclude trial 1 - it follows different convention for color
        to_excl = ((1:n_tr) == 1)';
                
        for tr = 1:n_tr            
            MotionStimulus1 = dat.MotionStimulus(tr,:);
            x_pix = MotionStimulus1{1};
            y_pix = MotionStimulus1{2};
            coherent_motion = MotionStimulus1{3};
            color = MotionStimulus1{4};
            t_sec = MotionStimulus1{5};
            color = color - 1; % Range [1, 2] -> [0, 1]
            
            if ismember(L.subjID, [2, 3]) && tr > 1
            end            
            
%             n_fr = size(x_pix, 1);
            n_fr = find(isnan(x_pix(:,1)), 1, 'first') - 1;
            x_pix = x_pix(1:n_fr, :);
            y_pix = y_pix(1:n_fr, :);
            coherent_motion = coherent_motion(1:n_fr, :);
            color = color(1:n_fr, :);
            t_sec = t_sec(1:n_fr, :);
            
            n_dot = size(x_pix, 2);
            t_fr = repmat((1:n_fr)', [1, n_dot]);
            
%             x_deg = x_pix / pix_per_deg;
%             y_deg = y_pix / pix_per_deg;

            rME{tr} = mean(coherent_motion, 2)' ...
                .* sign(dat.signMotionCoherence(tr));
            en_color_mean{tr} = (sum(color == 0, 2) - 2)';
            en_color_LLR{tr} = nanind(LLRNcol2, sum(color == 0, 2)' + 1);
            rCE{tr} = en_color_LLR{tr};
            
%             if any(color(:) < 0) || any(color(:) > 1)
%                 keyboard;
%                 error('Unexpected color!');
%             end
            
            xyct{tr} = [x_pix(:), y_pix(:), color(:), t_fr(:)];
        end
        tbl1.raw_en = [rME, rCE];
        tbl1.en_color_LLR = en_color_LLR;
        tbl1.en_color_mean = en_color_mean;
        tbl1.xyct = xyct;
        tbl1.to_excl = to_excl;
    
        %% Sanity check - strange - mean rCE across trial is bimodal!
%         % Plot motion & color coherence vs mean rME & rCE
%         mean_en = [
%             cellfun(@(v) nanmean(v(:)), rME), ...
%             cellfun(@(v) nanmean(v(:)), rCE)];
%         
%         fig_tag(sprintf('S%d_ND%d', i_subj, n_dim1));
%         clf;
%         for dim = 1:n_dim
%             subplot(n_dim, 1, dim);
%             plot(tbl1.cond(:,dim), mean_en(:,dim), 'ko')
%         end
        
        %%
        tbl = [tbl; tbl1];
    end    
end

%%
if exist(file_out, 'file')
    L = load(file_out);
    tbl0 = struct2table(L);
    for f = tbl.Properties.VariableNames(:)'
        tbl0.(f{1}) = tbl.(f{1});
    end
    fprintf('Updated following fields in %s : ', file_out);
    fprintf(' %s', tbl.Properties.VariableNames{:});
    fprintf('\n\n');

    fprintf('Did not update following fields in %s : ', file_out);
    vars_unchanged = setdiff( ...
        tbl0.Properties.VariableNames, ...
        tbl.Properties.VariableNames);
    fprintf(' %s', vars_unchanged{:});
    fprintf('\n\n');
    
    if ~inputYN(sprintf('Continue to save to %s (y/n)? ', file_out))
        return;
    else
        tbl = tbl0;
    end
end

%%
S_tbl = table2struct(tbl, 'ToScalar', true);
save(file_out, '-struct', 'S_tbl');
fprintf('Saved to %s\n', file_out);