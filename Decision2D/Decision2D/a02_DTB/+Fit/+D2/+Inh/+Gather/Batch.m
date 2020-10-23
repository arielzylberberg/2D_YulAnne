classdef Batch
methods
    function Pls = batch(B)
        subjs = Data.Consts.subjs_RT;
        for ii = 1:numel(subjs)
            subj = subjs{ii};
            
            try
                Pl = Fit.D2.Inh.Gather.PlotHeatmap( ...
                    'subj', subj);
            catch err
                warning(err_msg(err));
                continue;
            end 
            
            for figs = {'plot', 'plot_SNR_1', 'plot_SNR_1_drift_fac_0'}
                clf;
%                 try
                    Pl.(figs{1});
                    file = fullfile('Data', class(Pl), [subj '_' figs{1}]);
                    savefigs(file, 'size', [800 600]);
%                 catch err
%                     warning(err_msg(err));
%                 end
            end
            
            Pls{ii} = Pl;
        end
    end
    function plot_all(B, Pl)
        
    end
end
end