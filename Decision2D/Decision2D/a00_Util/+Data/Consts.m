classdef Consts
properties (Constant)
    subj_parad_all = {
        {'S1', 'RT'}
        {'S2', 'RT'}
        {'S3', 'RT'}
        {'S1', 'sh'}
        {'S2', 'sh'}
        {'S3', 'sh'}
        {'YK', 'sh'}
        }';
    subj_parad_RT = {
        {'S1', 'RT'}
        {'S2', 'RT'}
        {'S3', 'RT'}
        }';
    subj_parad_sh = {
        {'S1', 'sh'}
        {'S2', 'sh'}
        {'S3', 'sh'}
        {'YK', 'sh'}
        }';
    subjs_RT = {'S1', 'S2', 'S3', 'FR'};
    subjs_RT_incl_monk = {'S1', 'S2', 'S3', 'FR'};
    subjs_short = {'S1', 'S2', 'S3', 'YK'};
    subjs_sh = {'S1', 'S2', 'S3'}; % , 'YK'};
    
%     subj_parad_all = {
%         {'DX', 'RT'}
%         {'MA', 'RT'}
%         {'VL', 'RT'}
%         {'DX', 'sh'}
%         {'MA', 'sh'}
%         {'VL', 'sh'}
%         {'YK', 'sh'}
%         }';
%     subj_parad_RT = {
%         {'DX', 'RT'}
%         {'MA', 'RT'}
%         {'VL', 'RT'}
%         }';
%     subj_parad_sh = {
%         {'DX', 'sh'}
%         {'MA', 'sh'}
%         {'VL', 'sh'}
%         {'YK', 'sh'}
%         }';
%     subjs_RT = {'DX', 'MA', 'VL'};
%     subjs_RT_incl_monk = {'DX', 'MA', 'VL', 'FR'};
%     subjs_short = {'DX', 'MA', 'VL', 'YK'};
%     subjs_sh = {'DX', 'MA', 'VL', 'YK'};
%     
    subjs_strrep = {
        'DX', 'S1'
        'MA', 'S2'
        'VL', 'S3'
        };
    
    n_tr_initial_skip = 200;
    
    tasks = {'H', 'V'; 'A', 'A'};
    dimNames = {'M', 'C'};    
    dimNames_long = {'Motion', 'Color'};
    n_dim = 2;
    
    % en_per_coh(dim): used to transform momentary energy to coherence.
    % Get from Fit.D1.Common.DataChRtPdfEn.scale_en
%     en_per_coh = [711.4752, 112.2405];
    en_per_coh = ... % updated after BankRegr
        [
          1.000184e+01,  5.331142e+01
          1.278219e+01,  2.860860e+01
          5.378270e+01,  5.121885e+01
        ];
%         [
%         1 1
%         1 1
%         1 1
%         ];
%         [
%           1.322995e+02,  1.168342e+02
%           1.690765e+02,  6.269691e+01
%           7.114113e+02,  1.122482e+02
%         ];    
    data_root = '../Data';
    
    %% After fitting
    slprops0 = varargin2S({
        'RT', varargin2S({
            'min_sub', varargin2S({
                'S1', [0, 0]
                'S2', [0.1, 0.1]
                'S3', [0, 0.25]
                })
            'min_sup', varargin2S({
                'S1', [0.1, 1]
                'S2', [1, 0.1]
                'S3', [0.1, 1]
                })
            'min_sup15', varargin2S({
                'S1', [0.5, 1]
                'S2', [1, 0.5]
                'S3', [0.5, 1]
                })
            'min_sup125', varargin2S({
                'S1', [0.25, 1]
                'S2', [1, 0.25]
                'S3', [0.25, 1]
                })
            'min_sup0', varargin2S({
                'S1', [0.1, 1]
                'S2', [0, 1]
                'S3', [0, 1]
                })
            })
        })
    
    %%
    dt_frame = 1/75;
end
end