classdef PropFileNameTree < bml.oop.PropFileNameTree
    % PropFileNameTree
    % Set default root_data_dir
methods
    function PFile = PropFileNameTree(varargin)
        PFile = PFile@bml.oop.PropFileNameTree(varargin{:});
        PFile.root_data_dir = Data.Consts.data_root;
    end
end
end