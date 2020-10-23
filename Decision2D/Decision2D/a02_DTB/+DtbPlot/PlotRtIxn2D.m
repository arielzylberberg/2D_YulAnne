classdef PlotRtIxn2D < DtbPlot.PlotRt2D
properties
%     nSim = 1000;
%     
%     cond = [];
%     ch = [];
    
    E % Pred
end
methods
function [b, se, res] = get_y(Pl)
    % Return a scalar
    
    meanRTs = Pl.get_y@DtbPlot.PlotRt2D;
    
    x1 = meanRTs(1:(end-1),end);
    x2 = meanRTs(end,1:(end-1));
    [x1, x2] = ndgrid(x1,x2);
    X  = [x1(:), x2(:)];
    X  = bsxfun(@minus, X, mean(X));
    X  = [X, X(:,1) .* X(:,2)];
    
    y  = meanRTs(1:(end-1),1:(end-1));
    y  = y(:);
    
    res = glmwrap(X, y, 'normal');

    b  = res.b(4);
    se = res.se(4);
end
% function set_Pred(Pl, E)
%     assert(isa(E, 'Pred.PredDtb2D'));
%     Pl.E = E;
%     Pl.cond = E.cond;
%     Pl.ch = E.ch;
%     
%     if isempty(E.Td)
%         E.pred;
%     end
%     Pl.set_pdf(E.Td);
% %     Pl.conds = E.conds;
% %     Pl.pdfDat = E.TdDat;
% %     Pl.condFreq = E.get_condFreq;
% end
% function [b, se, bAll, seAll] = bootstrapGlm(Pl)
%     % Get X from the data
%     
%     error('Deprecated - simply using y from PlotRt2D');
%     
%     [condChs,~,dCondCh] = unique([Pl.cond, Pl.ch], 'rows');
%     nCondCh = length(condChs);
%     
%     error('replace cond with RTs!');
%     % replace cond with RTs % TODO
%     
%     for iDim = [2 1]
%         [~,~,condChs(:,iDim)] = unique(condChs(:,iDim));
%     end
%     
%     X = Pl.cond;
%     X = [X, X(:,1) .* X(:,2)]; % interaction term
%     nX = size(X, 2);    
%     
%     nTr  = size(Pl.cond, 1);
%     bAll = zeros(Pl.nSim, nX + 1);
%     seAll = zeros(Pl.nSim, nX + 1);
%     parfor iSim = 1:Pl.nSim
%         % Sample y
%         y = zeros(nTr, 1);
%         for iCond = 1:nCondCh
%             incl = dCondCh == iCond;
%             nIncl = nnz(incl);
%             
%             cCondCh = condChs(iCond,:);
%             y(incl) = randsample(Pl.nt, nIncl, true, ...
%                 Pl.pdf(:,cCondCh(1),cCondCh(2),cCondCh(3),cCondCh(4)));
%         end
%         
%         res = glmwrap(X, y, 'normal');
%         bAll(iSim,:)  = res.b(:)';
%         seAll(iSim,:) = res.se(:)';
%     end
%     
%     b  = median(bAll);
%     se(1,:) = prctile(bAll, normcdf(-1)*100);
%     se(2,:) = prctile(bAll, normcdf(1)*100);
% end
end
% methods (Static)
% function other2cond    
% %     for iDim = 2:-1:1
% %         [conds{iDim},~,dCond(:,iDim)] = unique(Pl.cond(:,iDim));
% %         nConds(iDim) = length(conds{iDim});
% %         
% %         for iCond = nConds(iDim):-1:1
% %             condFreq{iDim}(iCond) = nnz(conds{iDim}(iCond) == Pl.cond(:,iDim));
% %         end
% %     end
% %     [condsAllC{1}, condsAllC{2}] = ndgrid(num2cell(conds{1}), num2cell(conds{2}));
% %     condsAll = [[condsAllC{1}{:}], [condsAllC{2}{:}]];
% %     
% %     nCondsAll = size(condsAll, 1);
% %     condsFreqAll = zeros(nCondsAll, 1);
% %     for iCond = 1:nCondsAll
% %         condsFreqAll(iCond) = nnz(conds
% %     end
% end
% end
end