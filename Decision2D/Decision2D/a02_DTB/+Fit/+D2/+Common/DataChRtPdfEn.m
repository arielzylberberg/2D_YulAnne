classdef DataChRtPdfEn < Fit.D2.Common.DataChRtPdf
    % Fit.D2.Common.DataChRtPdfEn
    %
    % Has Ens that contains trial-by-trial fluctuation in the stimulus.
    % RT and Td_pred/obs_pdf_trs are added for trial-by-trial fit.
    % Their sizes are: t x tr x ch1 x ch2 (tr replaces cond1 x cond2)
    %
    % 2015 YK wrote the initial version.
properties (Transient, SetAccess = protected)
    Ens = {};
    RT_data_pdf_tr
    RT_pred_pdf_tr
    Td_pred_pdf_tr
end
properties
    refresh_rate = 75;
end
methods
    function Dat = DataChRtPdfEn(varargin)
        % See also: Fit.Common.DataChRtPdf

        Dat = Dat@Fit.D2.Common.DataChRtPdf(varargin{:});
        Dat.Ens = cell(1,2);
        Dat.add_deep_copy({'Ens'});
        Dat.set_Ens;
        Dat.set_Time;
    end
end
%% En, filtered by dat_filt
methods
    function ens = get_ens_cell(Dat, ts_args)
        % ens = get_ens_cell(Dat, ts_args)
        if nargin < 2, ts_args = {}; end
        ts_args = varargin2C(ts_args);
        
        for dim = 2:-1:1
            ens{dim} = Dat.get_en_cell(dim, ts_args);
        end
    end 
    function ens = get_ens_mat(Dat, ts_args)
        % ens = get_ens_mat(Dat, ts_args)
        if nargin < 2, ts_args = {}; end
        ts_args = varargin2C(ts_args);
        
        for dim = 2:-1:1
            ens{dim} = Dat.get_en_mat(dim, ts_args);
        end
    end 
    function en = get_en_cell(Dat, dim, ts_args)
        % en = get_en_cell(Dat, dim, ts_args)
        if nargin < 3, ts_args = {}; end
        ts_args = varargin2C(ts_args);
        
        en = Dat.Ens{dim}.get_ts_cell(ts_args{:});
        en = en(Dat.get_dat_filt);
    end
    function en = get_en_mat(Dat, dim, ts_args)
        % en = get_en_mat(Dat, dim, ts_args)
        if nargin < 3, ts_args = {}; end
        ts_args = varargin2C(ts_args);
        
        en = Dat.Ens{dim}.get_ts_mat(ts_args{:});
        en = en(Dat.get_dat_filt, :);
    end
end
%% Data loading and parsing
methods
    function load_data(Dat)
        if isempty(Dat.path)
            warning('path not set!');
            return; 
        end
        Dat.load_data@Fit.D2.Common.DataChRtPdf;
        for i_dim = Data.Consts.n_dim:-1:1 % Dat.get_n_dim:-1:1
            dim_name = Data.Consts.dimNames{i_dim}; % Dat.dimNames{i_dim};
            
            % Momentary energy
            en = Dat.get_ds0_field(['m', dim_name, 'E']); 
            
%             switch dim_name
%                 case 'C'
%                     % Smooth color
%                     en = cell2mat2(en);
%                     
%                     kernel = gampdf_ms((0:(1/Dat.refresh_rate):0.2)', ...
%                         0.05, 0.02, 1);
%                     en = conv_t(en', kernel)';
%             end
%             en = row2cell2(en);
%             
            % Cut en at RT
            n_fr_RT = Dat.Time.convert_sec2fr_ix( ...
                Dat.get_ds0_field('RT'));
                % = round(Dat.get_ds0_field('RT') * Dat.refresh_rate) + 1;
            
            tr_valid_RT = ~isnan(n_fr_RT);
            en(~tr_valid_RT) = {[]};
            
            for tr = hVec(find(tr_valid_RT))
                if i_dim == 1 && tr == 1136
                    disp(i_dim);
    %                 keyboard; % DEBUG
                end                
                
                dlen = length(en{tr}) - n_fr_RT(tr);
                if dlen < 0
%                     warning('en shorter than expected at tr=%d, dim=%d, dlen: %d\n', ...
%                         tr, i_dim, dlen);
%                     en{tr} = nan(1, n_fr_RT(tr));

%                     if dlen < -1 % May be the case when it is the excluded run
% %                         warning(['en shorter than expected at tr=%d, dim=%d: ' ...
% %                             'length(en{tr}): %d, n_fr_RT(tr): %d, dlen: %d\n'], ...
% %                             tr, i_dim, length(en{tr}), n_fr_RT(tr), dlen);
% %                         error('dlen < -1!');
%                     else
                        en{tr}((end+1):n_fr_RT(tr)) = nan; % 
%                     end
                end                    
                en{tr} = en{tr}(1:n_fr_RT(tr));
                nnan = nnz(isnan(en{tr}));
%                 if nnan > 0
%                     warning('en has NaN at tr=%d, dim=%d: #NaN: %d\n', ...
%                         tr, i_dim, nnan);
%                     if nnan > 1
%                         error('#NaN > 1!');
%                     end
%                 end
            end
            
            Dat.Ens{i_dim} = TimeAxis.TimeSeriesSorterInterpolable( ...
                Dat.Time, en);
        end
        f_all_nan = @(cc) cellfun(@(c) all(isnan(c)), cc);
        is_missing{1} = f_all_nan(Dat.ds.mME);
        is_missing{2} = f_all_nan(Dat.ds.mCE);
        for dim = 1:2
            is_missing1 = is_missing{dim};
            if any(is_missing1)
                n_missing = nnz(is_missing1);
                ds = Dat.ds;
                ds1 = ds(is_missing1, :);
                
                %%
                disp('-----');
                tabulate(ds1.condM)
                tabulate(ds1.condC)
                tabulate(ds1.task)
                crosstab(ds1.condM, ds1.condC)
                crosstab(ds1.task, ds1.condM)
                crosstab(ds1.task, ds1.condC)
                tabulate(ds1.i_all_Run);
                warning('missing en in dim %d, %d trials!', ...
                    dim, n_missing);
            end
        end
    end
    function set_ds0(Dat, ds0)
        % Omit ds0 to set a default dataset.
        if ~exist('ds0', 'var')
            Dat.set_ds0@Fit.D2.Common.DataChRtPdf;
            ds0 = copyFields(Dat.get_ds0, varargin2S({
                'mME', cell(0,1)
                'mCE', cell(0,1)
                }));
        end
        % - QUIRK -
        % Scale energy to approximately match coherence's scale.
        % Numbers are determined from get_ratio_En_cond().
%         ds0.mME = cellfun(@(v) v / 580, ...
%             row2cell2(ds0.mME, ds0.nME), ...
%             'UniformOutput', false);
%         ds0.mCE = cellfun(@(v) v / 0.18, ...
%             row2cell2(ds0.mCE, ds0.nCE), ...
%             'UniformOutput', false);
        Dat.set_ds0@Fit.D2.Common.DataChRtPdf(ds0);
    end
    function reset_RT_data_pdf(Dat)
        Dat.reset_RT_data_pdf@Fit.D2.Common.DataChRtPdf;
        Dat.RT_data_pdf_tr = [];
    end
    function ratio = get_ratio_En_cond(Dat, dim)
        % Give ratio such that approximately, En = cond * ratio.
        %
        % ratio = get_ratio_En_cond(Dat, dim)
        En = Dat.get_En(dim);
        En = En.get_ts_mat;
        
        cond = Dat.get_cond;
        cond = cond(:, dim);
        
        mean_En = nanmean(En, 2);
        
        ratio = regress(mean_En, cond);
    end
end
%% Condition-by-condition pdfs : summarize trial-by-trial pdfs
methods
    function v = get_RT_pred_pdf(Dat)
        % Summarize RT_pred_pdf_tr
        v = Dat.pdf_tr2cond(Dat.get_RT_pred_pdf_tr);
        
        % Don't normalize, since it is being averaged already 
        % (see pdf_tr2cond)
        % and we need to penalize against trials finished 
        % without being absorbed.
%         v = Dat.pdf_normalize_within_cond(v); 
    end
    function v = get_Td_pred_pdf(Dat)
        % Summarize Td_pred_pdf_tr
        v = Dat.pdf_tr2cond(Dat.get_Td_pred_pdf_tr);
%         v = Dat.pdf_normalize_within_cond(v);
    end
    function pdf_cond = pdf_tr2cond(Dat, pdf_tr)
        % pdf_cond = pdf_tr2cond(Dat, pdf_tr)
        
        if isempty(pdf_tr)
            pdf_cond = [];
            warning('Empty pdf_tr is given! Giving empty pdf_cond!');
            return;
        end
        
        dCond = Dat.get_dCond;
        pdf_cond = zeros(Dat.get_size_RT_Td_pdf);
        
        for dCond1 = 1:max(dCond(:,1))
            for dCond2 = 1:max(dCond(:,2))
                for ch1 = 1:2
                    for ch2 = 1:2
                        incl = (dCond(:,1) == dCond1) & (dCond(:,2) == dCond2);
                        pdf_cond(:,dCond1,dCond2,ch1,ch2) = ...
                            mean(pdf_tr(:,incl,1,ch1,ch2), 2);
                    end
                end
            end
        end
    end
%     function v = pdf_normalize_within_cond(Dat, v)
%         % Normalize within joint conditions using the original count
%         dCond = Dat.get_dCond;
%         nConds = Dat.get_nConds;
%         n_in_cond = permute(accumarray(dCond, 1, nConds, @sum), ...
%             [3, 1, 2]);
%         n_in_cond2 = sums(v, [Dat.dim_pdf.t, Dat.dim_pdf.ch]);
%         
%         disp(squeeze(n_in_cond - n_in_cond2));
%         
%         v = bsxfun(@rdivide, v, n_in_cond);
%     end
end
%% Trial-by-trial pdfs    
methods
    function v = get_size_RT_Td_pdf_tr(Dat)
        v = [Dat.get_nt, Dat.get_n_tr, 1, 2, 2];
    end
    
    function set_RT_data_pdf_tr(Dat, v)
        assert(isempty(v) || isequal(size(v), Dat.get_size_RT_Td_pdf_tr));
        Dat.RT_data_pdf_tr = v;
    end
    function v = get_RT_data_pdf_tr(Dat)
        if isempty(Dat.RT_data_pdf_tr)
            RT_ix = Dat.get_RT_ix;
            n_tr = Dat.get_n_tr;
            
            Dat.RT_data_pdf_tr = accumarray( ...
                [RT_ix, (1:n_tr)', ones(n_tr, 1), Dat.get_ch], ...
                1, ...
                Dat.get_size_RT_Td_pdf_tr, ...
                @sum);
        end
        if nargout > 0
            v = Dat.RT_data_pdf_tr;
        end
    end

    function set_RT_pred_pdf_tr(Dat, v)
        assert(isempty(v) || isequal(size(v), Dat.get_size_RT_Td_pdf_tr));
        Dat.RT_pred_pdf_tr = v;
    end
    function v = get_RT_pred_pdf_tr(Dat)
        v = Dat.RT_pred_pdf_tr;
    end

    function set_Td_pred_pdf_tr(Dat, v)
        assert(isempty(v) || isequal(size(v), Dat.get_size_RT_Td_pdf_tr));
        Dat.Td_pred_pdf_tr = v;
    end
    function v = get_Td_pred_pdf_tr(Dat)
        v = Dat.Td_pred_pdf_tr;
    end
end
%% Ens
methods
    function set_Ens(Dat, Ens)
        if exist('Ens', 'var')
            assert(iscell(Ens) && numel(Ens) == 2);
            assert(all( ...
                cellfun(@(c) isa(c, 'TimeAxis.TimeSeriesSorterInterpolable'), ...
                Ens)));
        else
            for ii = 1:2
                Ens{ii} = ...
                    TimeAxis.TimeSeriesSorterInterpolable(Dat.Time);
            end
        end
        Dat.Ens = Ens;
    end
    function En = get_En(Dat, i_dim)
        assert(isscalar(i_dim));
        assert(any(i_dim == [1 2]));
        En = Dat.Ens{i_dim};
    end
end
%% Time - unify with Ens's
methods
    function set_Time(Dat, Time)
        if exist('Time', 'var')
            Dat.set_Time@TimeAxis.TimeInheritable(Time);
        else
            Time = Dat.get_Time;
        end
        for ii = 1:numel(Dat.Ens)
            Dat.Ens{ii}.set_Time(Time);
        end
    end
end
%% Demo
methods (Static)
    function Dat = demo
        %%
        Dat = Fit.D2.Common.DataChRtPdfEn;
        Dat.set_path({}, 'A');
        Dat.load_data;
        %%
        disp(Dat);
        disp(Dat.Time);
        disp(Dat.Ens{1}.Time);
        disp(Dat.Ens{1}.get_Time_src);
    end
end
end