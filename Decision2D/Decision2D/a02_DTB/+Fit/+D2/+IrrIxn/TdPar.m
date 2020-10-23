classdef TdPar ...
        < Fit.D2.Bounded.TdPar ...
        & Fit.D2.IrrIxn.Td
    % Fit.D2.Bounded.TdPar
    %
    % Given two Td_pdfs, give a merged parallel Td_pdf.
    
    % 2015 YK wrote the initial version.
methods
    function W = TdPar
        W.set_Data;
    end
end
methods % (Static)
    function [Td_pdf, Td_last_pdf] = get_Td_pdf(W, Td_pdfs)
        % [Td_pdf, Td_last_pdf] = get_Td_pdf(W, Td_pdfs)
        %
        % Td_pdfs{dim}(t, cond1, cond2, ch1, ch2)
        % Td_pdf(t, dim, cond1, cond2, ch1, ch2)
        % Td_last_pdf(t, dim, cond1, cond2, ch1, ch2)
        
        assert(iscell(Td_pdfs) && numel(Td_pdfs) == 2);
        assert(isequal(sizes(Td_pdfs{1}, 1:3), sizes(Td_pdfs{2}, 1:3)));
        assert(size(Td_pdfs{1}, 4) == 2);
        assert(size(Td_pdfs{2}, 5) == 2);

        W.td_pdfs = Td_pdfs; % Cache

        %% Td_pdf
        for i_dim = 2:-1:1
            n_cond(i_dim) = size(Td_pdfs{i_dim}, 1 + i_dim);
        end
        for cond1 = n_cond(1):-1:1
            for cond2 = n_cond(2):-1:1
                for ch1 = 2:-1:1
                    for ch2 = 2:-1:1 
                        
                        p1 = Td_pdfs{1}(:,cond1,cond2,ch1,1);
                        p2 = Td_pdfs{2}(:,cond1,cond2,1,ch2);
                        cum_p1 = cumsum(p1);
                        cum_p2 = cumsum(p2);
                        p_max = p1 .* cum_p2 + p2 .* cum_p1 - p1 .* p2;
                        
%                         % The difference from non-ixn is that
%                         % both Td_pdfs are on both cond1 and cond2,
%                         % rather than either one, 
%                         % because there is interaction.
%                          [p_max, p_last] = ...
%                             bml.stat.max_distrib([
%                                 Td_pdfs{1}(:,cond1,cond2,ch1,1), ...
%                                 Td_pdfs{2}(:,cond1,cond2,1,ch2)
%                                 ]);
                               
                         Td_pdf(:,cond1,cond2,ch1,ch2) = p_max;
                         if nargout >= 2
                             error('Not supported!');
                             Td_last_pdf(:,:,cond1,cond2,ch1,ch2) = p_last;
                         end
                    end
                end
            end
        end
        
        %% DEBUG
        min_p_Td = min(Td_pdf(:));
        max_p_Td = max(Td_pdf(:));
        
        tol = 0;
        assert(min_p_Td >= -tol);
        assert(max_p_Td <= 1 + tol);
        Td_pdf = max(min(Td_pdf, 1 + tol), -tol);
        
        %% Normalize within each condition.
        % Since we observe only RT <= t_max,
        % we are effectively conditionalizing the RTs.
        Td_pdf = nan0(bsxfun(@rdivide, Td_pdf, sums(Td_pdf, [1, 4, 5])));
        
        %% DEBUG
        min_p_Td = min(Td_pdf(:));
        max_p_Td = max(Td_pdf(:));
        
        assert(min_p_Td >= -tol);
        assert(max_p_Td <= 1 + tol);
        Td_pdf = max(min(Td_pdf, 1 + tol), -tol);
    end
end
end