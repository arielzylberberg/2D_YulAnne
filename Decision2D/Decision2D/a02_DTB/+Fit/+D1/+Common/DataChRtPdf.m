classdef DataChRtPdf < Fit.Common.DataChRtPdf
    % Fit.D1.Common.DataChRtPdf
    %
    % 2015 YK wrote the initial version.
methods
function Dat = DataChRtPdf(varargin)
    % See also: FitData
    Dat.n_dim_task = 1;
    if nargin > 0
        Dat.init(varargin{:});
    end
end
function v = get_cond(Dat)
    v = Dat.get_ds_field(['cond' Dat.get_dim_rel_name]);
end
function v = get_conds_rel(Dat)
    v = Dat.get_conds;
    v = v{1};
end
function v = get_nConds_rel(Dat)
    v = Dat.get_nConds;
end
function v = get_ch(Dat)
    v = Dat.get_ds_field(['subj' Dat.get_dim_rel_name]);
end
function v = get_answer(Dat)
    v = Dat.get_ds_field(['corr' Dat.get_dim_rel_name]);
end
function v = get_RT(Dat)
    v = Dat.get_ds_field('RT');
end
function v = get_cond0(Dat)
    v = Dat.ds0.(['cond' Dat.get_dim_rel_name]);
end
function v = get_ch0(Dat)
    v = Dat.ds0.(['subj' Dat.get_dim_rel_name]);
end
function v = get_answer0(Dat)
    v = Dat.ds0.(['corr' Dat.get_dim_rel_name]);
end
function v = get_RT0(Dat)
    v = Dat.ds0.RT;
end
function s = get_dim_rel_name(Dat)
    ix = Dat.dim_rel;
    s = Data.Consts.dimNames{1, ix};
end
function S = get_dim_pdf(~)
    S = varargin2S({
        't', 1
        'cond', 2
        'ch', 3
        });
end
function S = get_dim_pdf_rel(Dat)
    S = Dat.get_dim_pdf;
end
function siz = get_size_RT_Td_pdf(Dat)
    siz = [Dat.Time.nt, Dat.get_nConds_rel, 2];
end
function demo_internal(Dat)
    Dat.set_path({}, 'H');
    Dat.load_data;
    disp(Dat);
end
end
methods (Static)
    function Dat = demo
        Dat = Fit.D1.Common.DataChRtPdf;
        Dat.demo_internal;
    end    
end
end