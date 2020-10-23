classdef PropFileNameTree < bml.oop.PropFileNameTree
    % PropFileNameTree
    % root_data_dir defaults to ../Data
methods
    function PFile = PropFileNameTree(varargin)
        PFile = PFile@bml.oop.PropFileNameTree(varargin{:});
        PFile.root_data_dir = '../Data';
    end
end
end