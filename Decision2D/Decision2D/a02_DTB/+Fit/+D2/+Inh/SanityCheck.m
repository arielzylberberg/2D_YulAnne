classdef SanityCheck < Fit.D2.Bounded.Main
properties
    tbl_models
    S0
end
methods
    function San = SanityCheck(varargin)
        C = {
            'model0', 'model1'
            'Ser',    'Ser'
            'Ser',    'InhSer'
            'Ser',    'InhSlice'
            'Ser',    'InhSliceFree'
            'Ser',    'InhEvScale'
            'Ser',    'Par'
            'InhSer', 'InhPar'
            'Par',    'Par'
            'Par',    'InhPar'
            'Par',    'InhSliceFree'
            'Par',    'InhEvScale'
            'Par',    'InhSer'
            };
        San.tbl_models = cell2table( ...
            C(2:end,:), ...
            'VariableNames', C(1,:));
        
        San.S0 = varargin2S(varargin, {
            'subj', 'DX'
            });
        varargin2props(San, San.S0);
    end
    function main(San)
        tbl = San.tbl_models;
        n = size(tbl, 1);
        
        %%
        if ~ismember('W1', tbl.Properties.VariableNames)
            tbl.W1 = cell(n, 1);
        end
        if ~ismember('cost1', tbl.Properties.VariableNames)
            tbl.cost1 = nan(n, 1);
        end
        
        %%        
        tbl = San.tbl_models;
        models_incl = 1:n; % [1:2, 6:7]; % [2, 6, 9] % 
        for ii = models_incl % 1:n % 2 % 9:10 % 4:5 % 1:n
            model0 = San.tbl_models.model0{ii};
            model1 = San.tbl_models.model1{ii};
            
%             if ~isempty(tbl.W1{ii,1})
%                 continue;
%             end
            
            C = varargin2C({
                'model', model1
                }, San.S0);
            W1 = San.create_RT(C{:});
            W1.Tnd.distrib = 'invgauss';
            
            switch model0
                case 'Ser'
                    switch model1
                        case 'InhSer'
                            W1.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                            W1.th.Dtb__drift_sigmaSq_fac_dim1_2 = 0;
                            W1.th.Dtb__drift_sigmaSq_fac_dim2_1 = 0;
                            W1.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                            for mu_name = { ...
                                    'mu_1_1', 'mu_1_2', 'mu_2_1', 'mu_2_2'}
                                W1.Tnd.th.(mu_name{1}) = 0.3 - W1.dt; 
                            end
                        case 'InhPar'
                            W1.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                            W1.th.Dtb__drift_sigmaSq_fac_dim1_2 = 1;
                            W1.th.Dtb__drift_sigmaSq_fac_dim2_1 = 1;
                            W1.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                        case 'InhSlice'
                        case 'InhSliceFree'
                            W1.th.Dtb__logit_slprop1 = -inf; % ...
%                                 W1.th_lb.Dtb__logit_slprop1;
                            W1.th.Dtb__logit_slprop2 = inf; % ...
%                                 W1.th_ub.Dtb__logit_slprop2;
                            for mu_name = { ...
                                    'mu_1_1', 'mu_1_2', 'mu_2_1', 'mu_2_2'}
                                W1.Tnd.th.(mu_name{1}) = 0.3 - W1.dt; 
                            end
                            
                        case 'InhEvScale'
                            W1.th.Dtb__logit_evscale1 = -inf; % ...
%                                 W1.th_lb.Dtb__logit_evscale1;
                            W1.th.Dtb__logit_evscale2 = inf; % ...
%                                 W1.th_ub.Dtb__logit_evscale2;
                            for mu_name = { ...
                                    'mu_1_1', 'mu_1_2', 'mu_2_1', 'mu_2_2'}
                                W1.Tnd.th.(mu_name{1}) = 0.3 - W1.dt; 
                            end
                    end
                case 'Par'
                    switch model1
                        case 'InhSer'
                            W1.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
                            W1.th.Dtb__drift_sigmaSq_fac_dim1_2 = 0;
                            W1.th.Dtb__drift_sigmaSq_fac_dim2_1 = 0;
                            W1.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                            for mu_name = { ...
                                    'mu_1_1', 'mu_1_2', 'mu_2_1', 'mu_2_2'}
                                W1.Tnd.th.(mu_name{1}) = 0.3 - W1.dt; 
                            end
                        case 'InhPar'
%                             for mu_name = { ...
%                                     'mu_1_1', 'mu_1_2', 'mu_2_1', 'mu_2_2'}
%                                 W1.Tnd.th.(mu_name{1}) = 0.3 + W1.dt; 
%                             end
%                             W1.th.Dtb__drift_sigmaSq_fac_dim1_1 = 1;
%                             W1.th.Dtb__drift_sigmaSq_fac_dim1_2 = 1;
%                             W1.th.Dtb__drift_sigmaSq_fac_dim2_1 = 1;
%                             W1.th.Dtb__drift_sigmaSq_fac_dim2_2 = 1;
                        case 'InhSliceFree'
                            W1.th.Dtb__logit_slprop1 = ...
                                inf; % W1.th_ub.Dtb__logit_slprop1;
                            W1.th.Dtb__logit_slprop2 = ...
                                inf;% W1.th_ub.Dtb__logit_slprop2;
                            
                        case 'InhEvScale'
                            W1.th.Dtb__logit_evscale1 = ...
                                inf; % W1.th_ub.Dtb__logit_evscale1;
                            W1.th.Dtb__logit_evscale2 = ...
                                inf; % W1.th_ub.Dtb__logit_evscale2;
                    end
            end
            
            cost1 = W1.get_cost;
            W1.get_Fl;
            
            tbl.W1{ii,1} = W1;
            tbl.cost1(ii,1) = cost1;
            San.tbl_models = tbl;
        end
        
        %%
        for ii = models_incl % 1:n
            ix0 = find(strcmp(tbl.model1, tbl.model0{ii}), 1, 'first');
            tbl.W0{ii,1} = tbl.W1{ix0,1};
            tbl.cost0(ii,1) = tbl.cost1(ix0,1);
        end
        
        tbl.dcost = tbl.cost1 - tbl.cost0;
        San.tbl_models = tbl;
        
        %%
        file = [fullfile('../Data', class(San), ...
            bml.str.Serializer.convert(varargin2S({
                'sbj', San.S0.subj
                'mdl', unique(tbl.model1)
                }))), '.txt'];
        if exist(file, 'file')
            delete(file);
        end
        mkdir2(fileparts(file));
        diary(file);
        disp(tbl(:, {'model0', 'model1', 'cost0', 'cost1', 'dcost'}));
        diary('off');
        
        %%
        for ii = models_incl % 1:n
            W0 = tbl.W0{ii};
            W1 = tbl.W1{ii};
            
            San.plot_p_comp(W0, W1, tbl.model0{ii}, tbl.model1{ii});           
        end
        
        %%
        S_file = tbl.W0{1}.S_file;
        S_file.mdl = unique(tbl.model1);
        file = fullfile('../Data', class(San), ...
            bml.str.Serializer.convert(S_file));
        save(file, 'San');
        fprintf('Saved to %s.mat\n', file);
    end
    function plot_p_comp(San, W0, W1, name0, name1)
        p0 = W0.Data.RT_pred_pdf;
        p1 = W1.Data.RT_pred_pdf;
        
        if ~exist('name0', 'var')
            name0 = 'W0';
        end
        if ~exist('name1', 'var')
            name1 = 'W1';
        end

        %%
        for conds = {
                [1,1]
                [1,5]
                [5,5]
                }'
            p0_plot = log10(p0(:,conds{1}(1),conds{1}(2),:,:));
            p1_plot = log10(p1(:,conds{1}(1),conds{1}(2),:,:));

            clf;
            plot(W1.t, reshape(p0_plot, W1.nt, []), 'k.');
            hold on;
            plot(W1.t, reshape(p1_plot, W1.nt, []), 'r.');
            hold off;

            title(sprintf('%s vs %s (cond%d,%d)', ...
                name0, ...
                name1, ...
                conds{1}(1), conds{1}(2) ...
                ));
            p_all = [p0_plot(:); p1_plot(:)];
            y_lim = [prctile(p_all, 1), ...
                     prctile(p_all, 99)];
            y_lim(2) = y_lim(1) + diff(y_lim) * 1.2;
            ylim(y_lim);
            ylabel('log(p_{RT})');
            xlabel('Time (s)');

            file = fullfile('../Data', class(San), ...
                bml.str.Serializer.convert(varargin2S({
                    'sbj', San.S0.subj
                    'mdl0', name0
                    'mdl1', name1
                    'cnd', conds{1}
                    })));
            savefigs(file);
        end
 end
    function [San, Ws] = load_demo(San)
        %%
        San = Fit.D2.Inh.SanityCheck;
        
        L = load('../Data_2D/Fit.D2.Inh.SanityCheck/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+dft=C+bnd=C+ssq=C+tnd=i+ntnd=4+msf=0+td=Ser+fsqs=1+fbst=1+mdl={InhEvScale,InhPar,InhSer,InhSlice,InhSliceFree,Par,Ser}.mat');
        San = L.San;
        Ser = San.tbl_models.W1{1};
        InhSer = San.tbl_models.W1{2};
        Par = San.tbl_models.W1{6};
        InhPar = San.tbl_models.W1{7};
        
        %%
        % Problem: the difference between Ser vs Par is big 
        % but that between InhSer vs InhPar is small.
        disp(L.San.tbl_models);
        
        %%
        Ws = packStruct(Ser, InhSer, Par, InhPar);
        
        %%
        for fs = fieldnames(Ws)'
            W1 = Ws.(fs{1});
            W1.get_Fl;
            
            figure;
            W1.Fl.runPlotFcns;
        end
        
        %%
        W = InhSer.Dtb;
        unabs = W.unabs_together_first{1};
        imagesc(unabs(:,:,5,5,2));
        td_1st = unabs(:,InhSer.y==0,5,5,2);
        plot(W.t, td_1st);
    end
end
end