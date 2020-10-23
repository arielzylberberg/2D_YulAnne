classdef FilterBankRegrSim < Fit.Filter.Motion.FilterBankRegr
methods
    function main_apply_filter_subj(Bank, varargin)
        S = varargin2S(varargin, {
            'parad', 'RT'
            'task', 'H'
            'ad_cond_incl', 1:2
            'truncate_st_fr', Bank.truncate_st_fr_find
            'truncate_en_fr', 0
            'crossval_ix', 0
            });
        C = S2C(S);
        Bank.load_beta(C{:});

        S = varargin2S(varargin, {
            'parad', 'RT'
            'task', 'H'
            'ad_cond_incl', 1:5
            'truncate_st_fr', 0
            'truncate_en_fr', 0
            'crossval_ix', 0
            'suffix', '_dot5s' % the only difference - load simulated 5s
            });
        C = S2C(S);
        
        Bank.load_dat(C{:});
        disp('-----');
        fprintf('Starting main_apply_filter to H:\n');
        disp(Bank.S0_file);
        disp('-----');
        Bank.main_apply_filter;   
        
        S = varargin2S(varargin, {
            'parad', 'RT'
            'task', 'A'
            'ad_cond_incl', 1:5
            'truncate_st_fr', 0
            'truncate_en_fr', 0
            'crossval_ix', 0
            'suffix', '_dot5s' % the only difference - load simulated 5s
            });
        C = S2C(S);
        
        Bank.load_dat(C{:});
        disp('-----');
        fprintf('Starting main_apply_filter to A:\n');
        disp(Bank.S0_file);
        disp('-----');
        Bank.main_apply_filter;       
    end
    function fr_en = get_fr_en(Bank)
        fr_en = 75 * 5 + zeros(size(Bank.rt_fr));
    end
end
end