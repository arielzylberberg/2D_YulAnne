classdef EvTimeD2 < ...
        IxnKernel.EvTime.EvTimeD1
properties
    Evs % {dim} = IxnKernel.EvTime.EvTimeD1
    n_dim = 2;
end
% properties (Dependent) % Inherited from EvTimeD1
%     ch % (trial, 1, dim)
%     ev % (trial, fr, dim)
%     noise_internal % (trial, fr, dim) % Unknown to experimenter
%     td_fr % (trial, 1) % frame (= time bin) of the time of first crossing
%     rt_fr % (trial, 1) % frame (= time bin) of the time of report
% end
methods
    function Ev = EvTimeD2(varargin)
        Ev.Evs = {IxnKernel.EvTime.EvTimeD1, IxnKernel.EvTime.EvTimeD1};
        if nargin > 0
            Ev.init(varargin{:});
        end
    end
    function init(Ev, varargin)
        varargin2props(Ev, varargin, true);
        for dim = 1:Ev.n_dim
            varargin2props(Ev.Evs{dim}, varargin, true);
        end
    end
end
methods
    function v = get_ch(Ev)
        v = cat(3, Ev.Evs{1}.ch, Ev.Evs{2}.ch);
    end
    function set_ch(Ev, v)
        Ev.Evs{1}.ch = v(:,:,1);
        Ev.Evs{2}.ch = v(:,:,2);
    end
    
    function v = get_ev(Ev)
        v = cat(3, Ev.Evs{1}.ev, Ev.Evs{2}.ev);
    end
    function set_ev(Ev, v)
        Ev.Evs{1}.ev = v(:,:,1);
        Ev.Evs{2}.ev = v(:,:,2);
    end
    
    function v = get_noise_internal(Ev)
        v = cat(3, Ev.Evs{1}.noise_internal, Ev.Evs{2}.noise_internal);
    end
    function set_noise_internal(Ev, v)
        Ev.Evs{1}.noise_internal = v(:,:,1);
        Ev.Evs{2}.noise_internal = v(:,:,2);
    end
    
    function v = get_td_fr(Ev)
        % use Evs{1}. Ignore Evs{2}.
        v = Ev.Evs{1}.td_fr;
    end
    function set_td_fr(Ev, v)
        Ev.Evs{1}.td_fr = v;
        Ev.Evs{2}.td_fr = v;
    end
    
    function v = get_rt_fr(Ev)
        % use Evs{1}. Ignore Evs{2}.
        v = Ev.Evs{1}.rt_fr;
    end
    function set_rt_fr(Ev, v)
        Ev.Evs{1}.rt_fr = v;
        Ev.Evs{2}.rt_fr = v;
    end
end
%% Import
methods
    function import_data(Ev, file0, varargin)
        % Import data
        % from what's exported by Fit.D2.Common.DataFilterEn.export_data.
        %
        % import_data(Ev, file0, varargin)
        %
        % Required variables in the file:
        % ch(tr, dim)
        % cond(tr, dim)
        % ens_mat{dim}(tr, fr)
        % rt_fr(tr, 1)
        %
        % Options
        % -------
        % 'dif_rel_incl', 'all'
        % 'dif_irr_incl', 'all'
        
        S = varargin2S(varargin, {
            'dif_incl', {1:3,1:3} % {1,1} % {'all','all'}
            'st_fr', 11
            });
        
        %%
        fprintf('Importing data from %s\n', file0);
        L = load(file0);
        
        %% Import S_file
        try
            [~, nam] = fileparts(file0);
            S_file = bml.str.Serializer.convert(nam);
            Ev.S_file_ = S_file;
        catch
        end
        
        %%
        n_dim = size(L.cond, 2);
        for dim = n_dim:-1:1
            [~,~,ad_cond(:,dim)] = unique(abs(L.cond(:,dim)));
        end
        n_tr = size(ad_cond, 1);
        
        tr_incl = true(n_tr, 1);
        for dim = 1:n_dim
            if ~isequal(S.dif_incl{dim}, 'all')
                tr_incl = tr_incl & ...
                    bsxEq(ad_cond(:,dim), ...
                        S.dif_incl{dim});
            end
        end
        Ev.S_file_.dfr = S.dif_incl;
        Ev.S_file_ = bml.struct.rmfield(Ev.S_file_, {
            'dim_r', 'dif_i', 'acc_i',' dfr', 'dfi', 'aci'
            });
            
        %%
        for dim = 1:numel(L.ens_mat)
            Ev.Evs{dim}.ch = L.ch(tr_incl, dim) == 2;
            Ev.Evs{dim}.ev = L.ens_mat{dim}(tr_incl, S.st_fr:end);
            Ev.Evs{dim}.cond = L.cond(tr_incl, dim);
            Ev.Evs{dim}.rt_fr = L.rt_fr(tr_incl);
        end
        
        Ev.td_fr = L.rt_fr(tr_incl);
        Ev.rt_fr = L.rt_fr(tr_incl);
    end
end
end