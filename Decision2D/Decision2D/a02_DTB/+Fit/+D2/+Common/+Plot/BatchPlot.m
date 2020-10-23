classdef BatchPlot < Fit.Common.Plot.BatchPlot
    % Fit.D2.Common.Plot.BatchPlot
    %
    % 2016 YK wrote the initial version.
methods
    function fig_files = Fl2figs(~, Fl, S_fit, pth_fig, varargin)
        opt = varargin2S(varargin, {
            'skip_existing_fig', true
            });
        
        S2s = bml.str.Serializer;
        Plt = [];
        fig_files = {};
        
        for fig_batch = {
                'plotfuns', {}
                'rt', {'dimOnX', 1}
                'rt', {'dimOnX', 2}
                'ch', {'dimOnX', 1}
                'ch', {'dimOnX', 2}
                'rt_log', {'dimOnX', 1}
                'rt_log', {'dimOnX', 2}
                'ch_log', {'dimOnX', 1}
                'ch_log', {'dimOnX', 2}
                }'
            [fig, opt_fig] = deal(fig_batch{:});
            
            S_fig = varargin2S({
                'fig', fig
                }, S_fit);
            S_fig = varargin2S(opt_fig, S_fig);
            
            file_fig = fullfile(pth_fig, S2s.convert(S_fig));

            if exist([file_fig '.fig'], 'file') ...
                    && opt.skip_existing_fig
                fprintf('--- Skipping existing figure %s\n', ...
                    [file_fig '.fig']);
            else
                if isempty(Plt)
                    Plt = eval(bml.pkg.get_class_rel(Fl.W, 'Plot'));
                    Plt.set_Fl(Fl);
                end
                
                clf;
                Plt.(fig)(S_fig);

                mkdir2(fileparts(file_fig));
                
                % size is set in Plt for each figure.
                savefigs(file_fig, 'size', []);
                
                fig_files = [fig_files; {file_fig}]; %#ok<AGROW>
            end
        end
    end
end
end