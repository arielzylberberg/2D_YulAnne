classdef BatchLogistic < Fit.Common.Plot.BatchPlot
    % Fit.D2.Common.Plot.BatchLogistic
    %
    % 2016 YK wrote the initial version.
methods
    function tableFile2logistic(BPlt, tab_file, varargin)
        if ~exist('tab_file', 'var') || isempty(tab_file)
            tab_file = uigetfile2('Data/*.mat');
        end
        L = load(tab_file, 'ds');
        BPlt.Fl2logistic('Fl_files', L.ds.file_fit, varargin{:});
    end
    function Fl2logistic(~, varargin)
        % Could use data files instead of Fl,
        % but in the future, may plot predictions, too,
        % in which case Fl (or W) is necessary.
        
        %%
        S_batch = varargin2S(varargin, {
            'Fl_files', {}
            'plot', {'LogisticSlope', ...
                     'LogisticBias', ...
                     'LogisticThres', ...
                     'LogisticSPE'}
            'Plt_args', {{'dimOnX', 1}, {'dimOnX', 2}}
            });
        
        if isempty(S_batch.Fl_files)
            S_batch.Fl_files = uigetfile2;
            if isempty(S_batch.Fl_files)
                fprintf('Aborted by user!\n');
                return;
            end
        end
        
        [Ss, n_Ss] = factorizeS(S_batch);
        
        S2s = bml.str.Serializer;
        
        n = numel(S_batch.Fl_files);
        for ii = 1:n
            Fl_file = S_batch.Fl_files{ii};
            L = load(Fl_file);
            Fl = L.Fl;
            
            [pth0, nam0] = fileparts(Fl_file);
%             pth0_aft_Data = pth0( ...
%                 max(bml.str.strfind_end(pth0, 'Data/')):end);
%             pth = fullfile('Data', class(BPlt), pth0_aft_Data);
            pth_fig = fullfile(fileparts(pth0), 'fig');
            
            for jj = 1:n_Ss
                S = Ss(jj);
                
                clf;
                Fit.D2.Common.Plot.(S.plot)(Fl, S.Plt_args{:});
                bml.plot.beautify;
                grid on;
                
                S_file = S;
                S_file = copyFields(S_file, varargin2S(S_file.Plt_args));
                S_file = rmfield(S_file, {'Fl_files', 'Plt_args'});
                
                nam = S2s.str_con(nam0, S2s.convert(S_file));
                file_fig = fullfile(pth_fig, nam);
                
                bml.plot.title(nam);
                if S_file.dimOnX == 1
                    xlabel('Motion coherence');
                    if any(strfinds(S.plot, {'Thres', 'SPE'}))
                        ax = gca;
                        ylabel([ax.YLabel.String, ' (Color coherence)']);
                    end
                else
                    xlabel('Color coherence');
                    if any(strfinds(S.plot, {'Thres', 'SPE'}))
                        ax = gca;
                        ylabel([ax.YLabel.String, ' (Motion coherence)']);
                    end
                end
                
                savefigs(file_fig);
            end
        end
    end
end
end