classdef DataLocator < matlab.mixin.Copyable
% Data.DataLocator
    
% 2015 YK wrote the initial version.
methods (Static)
function [files, S] = pred(varargin)
    % [files, S] = pred(...)
    % 
    % format: Data_2D/class/subdir/model2D_parad_subjext
    %
    %     'class', 'PredMetaInh'
    %     'model2D', 'ser'
    %     'parad', 'RT'
    %     'subj', 'VL'
    %     'postfix', ''
    %     'ext', '.mat'
    S = varargin2S(varargin, {
        'class', 'PredMetaInh'
        'subdir', ''
        'model2D', 'ser'
        'parad', 'RT'
        'subj', 'VL'
        'task', 'A'
        'postfix', ''
        'ext', '.mat'
        });
    files = Data.DataLocator.factorize('../Data_2D/%s/%s_%s_%s%s%s', ...
        fullfile(S.class, S.subdir), S.model2D, S.parad, S.subj, S.postfix, S.ext);
end
function [files, S, Ss, n] = sTr(varargin)
    % [files, S, Ss, n] = sTr(varargin)    
    % 
    % format: ../Data_2D/subdir/parad_SubjPostfixExt
    %
    %     'subdir', ''
    %     'parad', 'RT'
    %     'subj', 'VL'
    %     'postfix', ''
    %     'ext', '.mat'
    S = varargin2S(varargin, {
        'subdir', ''
        'parad', 'RT'
        'subj', 'VL'
        'task', 'A'
        'postfix', ''
        'ext', '.mat'
        });    
    
    [Ss, n] = bml.args.factorizeS(S);
    
    files = arrayfun(@(S) fullfile('../Data_2D/sTr', S.subdir, ...
        sprintf('%s_%s%s%s', S.parad, S.subj, S.postfix, S.ext)), ...
        Ss, 'UniformOutput', false);
    
%     files = Data.DataLocator.factorize('../Data_2D/%s/%s_%s%s%s', ...
%         fullfile('sTr', S.subdir), S.parad, S.subj, S.postfix, S.ext);
end

%% Common utility
function s = factorize(fmt, varargin)
    % s = factorize(fmt, [char_or_cell_array1, ...])
%     n_fac = length(varargin);
%     for i_fac = 1:n_fac
%         if ~iscell(varargin{i_fac})
%             varargin{i_fac} = varargin(i_fac); % enforce cell factor.
%         end
%     end
    [c, n] = bml.args.factorize(varargin);
    s = cell(n, 1);
    for ii = 1:n
        s{ii} = sprintf(fmt, c{ii,:});
    end
end
end
end