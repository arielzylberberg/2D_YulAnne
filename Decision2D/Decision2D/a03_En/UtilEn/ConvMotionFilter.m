classdef ConvMotionFilter < handle
	% Filter that PsyRDKCol uses to calculate momentary motion energy.
    %
    % .apply    : modify!! consider pow2(nextpow2(length1 + length2 - 1))
	%
	% filt		= psyMotionFilter('degPerPix', ~, 'dt', ~)
    %
    % Adapted from NYS's version. Optimized & wrapped into a class by YK.
    % 2019 renamed from PsyMotionFilter to ConvMotionFilter;
    %      interface with bwPix and xyct rather than RDKCol
	
	properties
        % degPerPix: assuming monitor width of 800 pix, 0.35m; 
        %            and viewing distance of 0.58m.
        %            as in Shadlen lab (UW) psychophysics rig.
        %            Given RDKCol, adapts appropriately.
		degPerPix	= 0.1319;	
        dt			= 1/75; 
		
		ordX	= 4;
		sigX	= 0.35/3; % degree % NYS: 0.5 -- differ from Kiani 2008..
                        % Controls both envelope (deg) & slant (deg/sec).
                        % 0.35/3 because I used 5/3 deg/sec.
		sigY	= 0.08;	% degree
		n1		= 3;	% after Kiani et al., 2008
		n2		= 5;	% after Kiani et al., 2008
		k		= 60;	% after Kiani et al., 2008
	
        dotSizePix = 2;
        apRPix  = [];
        
        lenX    = 75;   % 10 deg. Arbitrary value to help testing.
		lenY    = 75;   % 10 deg. Arbitrary value to help testing.
		lenT    = 60;   % 60 fr.  Arbitrary value to help testing.
		
		xVec
		yVec
		tVec
        
        addSec  = 0.2;
    end
    
    properties (Transient)
        % filters in the time domain
		filt = struct('left1', [], 'left2', [], 'right1', [], 'right2', []);
    end
    
    properties (Dependent)
        addFr
    end
	
	methods
        function demo(me)
            o = load('RDKCol');
            o.RDKCol.Scr = o.Scr;
            Filt0 = ConvMotionFilter(o.RDKCol);
            
            %%
            RDKCol.getCPix;
            tic; 
            mE0 = Filt0.apply(RDKCol); 
            plot(mE0);
            toc;
        end
		function me = ConvMotionFilter(RDKCol, varargin)
            % me = ConvMotionFilter(RDKCol, varargin)
            
            % If RDKCol is provided, use it to specify relevant properties.
            if nargin>=1 && ~isempty(RDKCol)
                me.degPerPix = 1/RDKCol.Scr.info.pixPerDeg;
                me.dt        = 1/RDKCol.Scr.info.refreshRate;
                
                if isempty(RDKCol.bwPix)
                    RDKCol.getCPix;
                end
                
                me.lenX	= round(size(RDKCol.bwPix, 2)); %  *2 - 1);
                me.lenY	= round(size(RDKCol.bwPix, 1)); %  *2 - 1);
                
            else
                % Otherwise, set each property manually. If not, use defaults.
                varargin2fields(me, varargin);
                S = varargin2S(varargin, {
                    'apRPix', []
                    'dotSizePix', 2
                    });
                if ~isempty(S.apRPix)
                    xyLen = me.get_xyLen(S.dotSizePix, S.apRPix);
                    me.lenX = xyLen;
                    me.lenY = xyLen;
                end                
            end
            
            me.lenT	= find(abs(me.tempImpResp(me.n2, me.k, ...
                                            0.15:me.dt:1)) ...
                                       < 1e-5, 1, 'first') ...
                                  + ceil(0.15/me.dt) + 1;
			
            me.init;
        end
        function init(me)
            % x/y/tVec
			me.xVec	= linspace(-(me.lenX-1)/2*me.degPerPix, ...
									(me.lenX-1)/2*me.degPerPix, me.lenX);
			me.yVec	= linspace(-(me.lenY-1)/2*me.degPerPix, ...
									(me.lenY-1)/2*me.degPerPix, me.lenY);
			me.tVec	= linspace(0, ...
								  (me.lenT-1)*me.dt, me.lenT);
							
			% temporal filters
			tFast	= me.tempImpResp(me.n1, me.k, me.tVec);
			tSlow	= me.tempImpResp(me.n2, me.k, me.tVec);

			% spatial filters along x
			[xEven,xOdd] = me.cauchy(me.xVec, me.sigX, me.ordX);

			% spatial filter along y
			yFilt	= exp(-me.yVec.^2/(2*me.sigY^2));

			% spatial filter: (y, x)
			yxEven	= yFilt' * xEven;
			yxOdd	= yFilt' * xOdd;

			% spatiotemporal filter: (y, x, t)
			fastOdd	= repmat(yxOdd, [1 1 length(me.tVec)]) ...
				   .* repmat(permute(tFast, [1 3 2]), ...
                            [length(me.xVec), length(me.yVec)]);

			fastEven= repmat(yxEven, [1 1 length(me.tVec)]) ...
				   .* repmat(permute(tFast, [1 3 2]), ...
                            [length(me.xVec), length(me.yVec)]);

			slowOdd = repmat(yxOdd, [1 1 length(me.tVec)]) ...
				   .* repmat(permute(tSlow, [1 3 2]), ...
                            [length(me.xVec), length(me.yVec)]);

			slowEven= repmat(yxEven, [1 1 length(me.tVec)]) ...
				   .* repmat(permute(tSlow, [1 3 2]), ...
                            [length(me.xVec), length(me.yVec)]);

			% final
			me.filt.left1	= Psy3DFilter(slowEven + fastOdd);
			me.filt.left2	= Psy3DFilter(fastEven - slowOdd);

			me.filt.right1	= Psy3DFilter(slowEven - fastOdd);
			me.filt.right2	= Psy3DFilter(fastEven + slowOdd);
        end
        function clearFilt(me)
            for cField = {'left1', 'left2', 'right1', 'right2'}
                delete(me.filt.(cField{1}));
            end
            me.filt = struct('left1', [], 'left2', [], 'right1', [], 'right2', []);
        end
        function h = plot_filt(me, ax, varargin)
            % h = show(me, ax)
            
            S = varargin2S({
                'x0', 0
                'y0', 0
                't0', 3/75
                });
            
			h.fig = gcf;
			if nargin<2, ax = 'tx'; end
            
            unit = varargin2S({
                'x', '(deg)'
                'y', '(deg)'
                't', '(sec)'
                });
			
			iField = 0;
			for cField = fieldnames(me.filt)'
				iField = iField + 1;
				
                switch ax
					case 'tx'
                        v0 = S.y0;
						[~, iY]	= min(abs(me.yVec - v0));
						cSlice = squeeze(me.filt.(cField{1}).reg(iY,:,:));
					case 'ty'
                        v0 = S.x0;
						[~, iX]	= min(abs(me.xVec - v0));
						cSlice = squeeze(me.filt.(cField{1}).reg(:,iX,:));
					case 'xy'
                        v0 = S.t0;
						[~, iT]	= min(abs(me.tVec - v0));
						cSlice = squeeze(me.filt.(cField{1}).reg(:,:,iT));						
                end
				
                v_plot = cell(1, 2);
                for dim = 1:2
                    o_dim = 3 - dim;
                    
                    switch ax(dim)
                        case 't'
                            v = ((1:size(cSlice, o_dim)) - 1) * me.dt;
                            
                        case {'x', 'y'}
                            center = (size(cSlice, o_dim) - 1) / 2;
                            v = (-center:center) * me.degPerPix;
                    end
                    v_plot{dim} = v;
                end
                
				subplot(2,2,iField);
				h.(cField{1}) = imagesc( ...
                    v_plot{1}, v_plot{2}, ...
                    abs(cSlice));
%                     max(log10(abs(cSlice)), -3));
				title(cField{1});
				xlabel([ax(1), ' ', unit.(ax(1))]);
				ylabel([ax(2), ' ', unit.(ax(2))]);
                
                hold on;
                switch ax
                    case {'tx'}
                        for deg_per_sec = [-5, 0, 5] % -5/3, 0, 5/3, 5]
                            x = v_plot{dim};
                            y = x * deg_per_sec;
                            plot(x, y, 'w:');
                        end
                        uistack(crossLine('v', S.t0, 'w:'), 'top');
                        
                        [~, dpix_per_pair, dt_coherent] ...
                            = me.get_xyct_impulse();
                        deg_per_pair = dpix_per_pair * me.degPerPix;
                        plot([0, dt_coherent], [0, deg_per_pair], 'wo');
                        plot([0, dt_coherent], [0, -deg_per_pair], 'wo');                        
                        
                    case {'ty'}
                        uistack(crossLine('h', S.x0, 'w:'), 'top');
                        uistack(crossLine('v', S.t0, 'w:'), 'top');
                    case {'xy'}
                        axis square
                        uistack(crossLine('h', S.y0, 'w:'), 'top');
                        uistack(crossLine('v', S.x0, 'w:'), 'top');
                end
                hold off;
			end
			
% 			colorbar;
        end        
		function mE = apply(me, src, varargin) 
            % mE = apply(me, bwPix_or_RDKCol, ...)
            %
            % bwPix(x_pix, y_pix, frame)
            % RDKCol: PsyRDKCol
            %
            % OPTIONS:            
            % 'addSec', me.addSec % add seconds at the end
            % 'input', 'xyct'
            
            %
            % See also PsyRDKCol.EnMot
            
            S = varargin2S(varargin, {
                'addSec', me.addSec % add seconds at the end
                'input', 'xyct' % 'xyct'|'bwPix'|'RDK'
                'dotSizePix', me.dotSizePix
                'apRPix', me.apRPix
                'use_fft', false
                });

            if isempty(me.filt.left1), me.init; end
            addSec = S.addSec;
            
            switch S.input
                case 'RDK'
                    bwPix = src.bwPix;
                case 'xyct'
                    bwPix = me.xyct2full(src);
                case 'bwPix'
                    bwPix = src;
                otherwise
                    error('Unknown S.input = %s\n', S.input);
            end
            
            cTLen  = size(bwPix,3) ...
                   + round(addSec / me.dt);
            
            sizOrig = [sizes(bwPix, [1 2]), cTLen];
			
            cTLen  = pow2(nextpow2(cTLen));

            cBWPix = cat(3, bwPix, ...
                            zeros([sizes(bwPix,[1 2]), ...
                                cTLen - size(bwPix,3)]));
               
%             resp.left1 = real(ifftn( fStim .* me.filt.left1.f(cTLen)));
%             resp.left2 = real(ifftn( fStim .* me.filt.left2.f(cTLen)));
%             resp.right1 = real(ifftn( fStim .* me.filt.right1.f(cTLen)));
%             resp.right2 = real(ifftn( fStim .* me.filt.right2.f(cTLen)));

            if S.use_fft
                % Optimize MATLAB's FFT.
                %   Takes a long time (~several seconds) on the first run, but
                %   improves performance throughout the current matlab session.
                fftw('planner', 'patient');				

                fStim	= fftn(cBWPix);
                for cFields = fieldnames(me.filt)'
                    cField = cFields{1};
    				resp.(cField) = real(ifftn( fStim .* me.filt.(cField).f(cTLen)));
                end
            else
                for cFields = fieldnames(me.filt)'
                    cField = cFields{1};

                    resp.(cField) = convn(cBWPix, me.filt.(cField).reg, 'full'); %, 'same');
                end
            end
            			
			resp.right	= sqrt(resp.right1 .^2	+ resp.right2 .^2);
			resp.left	= sqrt(resp.left1 .^2	+ resp.left2 .^2);
			
			mE = squeeze(sum(sum(resp.right - resp.left, 1),2));
            
            mE = mE(1:sizOrig(3));
        end
        function v = get.addFr(me)
            v = me.addSec / me.dt;
        end
        function cPix = xyct2full(me, xyct, varargin)
            % cPix = xyct2full(xyct, varargin)
            %
            % dotSizePix, apRPix: scalar in pix.
            % cPix(y, x, fr, color)
            % bwPix(y, x, fr)
            %
            % OPTIONS:
            % 'useGPU', false
            % 'dotSizePix', me.dotSizePix
            % 'apRPix', me.apRPix
            % 'use_color', false
            
            S = varargin2S(varargin, {
                'use_GPU', false
                'dotSizePix', me.dotSizePix
                'apRPix', me.apRPix
                'use_color', false
                });
            dotSizePix = S.dotSizePix;
            apRPix = S.apRPix;
            
            % Get x, y, color, time
            if isempty(xyct), cPix = []; bwPix = []; return; end
            
            [xyLen, pixDot] = ConvMotionFilter.get_xyLen(dotSizePix, apRPix);

            xyct(:,1:2) = round(xyct(:,1:2) + xyLen / 2 + 0.5);
%             xyct(:,1:2) = round(bsxfun(@plus, xyct(:,1:2), apRPix(1))) + 1;
            
            tLen    = max(xyct(:,4));

%             ix      = sub2ind([xyLen, xyLen, tLen, 2], xyct(:,2), xyct(:,1), xyct(:,4), xyct(:,3)+1);
%             ccPixOrig = zeros(xyLen, xyLen, tLen, 2);
%             ccPixOrig(ix) = 1;
            xyct(:,3) = xyct(:,3) + 1;
            
            if S.use_color
                cPix = zeros(xyLen, xyLen, tLen, 2);
                for xPixDot = pixDot
                    for yPixDot = pixDot
                        cPix = cPix + accumarray( ...
                            xyct(:,[2,1,4,3]) + [xPixDot, yPixDot, 0, 0], ...
                            1, [xyLen, xyLen, tLen, 2], @sum);
                    end
                end
            else
                cPix = zeros(xyLen, xyLen, tLen);
                for xPixDot = pixDot
                    for yPixDot = pixDot
                        cPix = cPix + accumarray( ...
                            xyct(:,[2,1,4]) + [xPixDot, yPixDot, 0], ...
                            1, [xyLen, xyLen, tLen], @sum);
                    end
                end
            end            
            
            % Consider type
            if S.use_GPU
                try
                    cPix  = gpuArray(cPix);
                catch 
                end
            end
        end        
        function plot_xyct(me, xyct, varargin)
            S = varargin2S(varargin, {
                'fr', 1
                });
            bwPix = me.xyct2full(xyct);

            xy_pix = (1:me.lenX) - (1 + me.lenX) / 2;

            imagesc(xy_pix, xy_pix, bwPix(:,:,S.fr));
            hold on;
            axis equal;
            xy_lim = xy_pix([1, end]) + [-0.5, +0.5];
            xlim(xy_lim);
            ylim(xy_lim);
            
            dots_incl = xyct(:,4) == S.fr;
            xyct1 = xyct(dots_incl, :);
            plot(xyct1(:,1), xyct1(:,2), 'wo');
            
            title(sprintf('Frame %d', S.fr));
            xlabel('x (pix)');
            ylabel('y (pix)');
        end
        function [xyct, dpix_per_pair, dt_coherent] ...
                = get_xyct_impulse(me, varargin)
            S = varargin2S(varargin, {
                'dfr', 3
                'direction', 1
                'deg_per_sec', 5 * (me.sigX / 0.35)
                'dt', me.dt
                });
            dt_coherent = S.dfr * S.dt;
            dpix_per_pair = ...
                S.direction * S.deg_per_sec * dt_coherent / me.degPerPix;
            
            center = 0;
%             center = 0.5 + S.direction;
            
            xyct = [
                center,            0, 1, 1
                center + dpix_per_pair, 0, 1, 1 + S.dfr
                ];
        end
        function [h, ME, t] = plot_impulse(me, varargin)
            [ME, t] = me.get_impulse_response(varargin{:});
            h = plot(t, ME, 'k-');
        end
        function [ME, t] = get_impulse_response(me, varargin)
            xyct = me.get_xyct_impulse(varargin{:});            
            ME = me.apply(xyct);
            t = ((1:length(ME)) - 1) * me.dt;
        end
    end
    
	
	methods (Static)	
		function time_response = tempImpResp(n,k,t)
			% time_response = temp_imp_resp(n,k,t)
			%
			% Produces a temporal impulse response function using the from from
			% figure 1 in Adelson & Bergen (1985)
			%
			% It's pretty much a difference of Poisson functions with different
			% time constants.
            %
			% Geoff Boynton wrote this at CSH '95

			time_response=(k*t).^n .* exp(-k*t).*(1/factorial(n)-(k*t).^2/factorial(n+2));
        end				
		function [xEven xOdd] = cauchy(x, sigX, ordX)
            % [xEven xOdd] = cauchy(x, sigX, ordX)
            %
			% Kiani et al., 2008
			%
			% transcribed by NYS, then by Y Kang, 2011.

			alpha	= atan2(x, sigX);
			xCommon = cos(alpha).^ordX;
			xEven	= xCommon .* cos(ordX .* alpha);
			xOdd	= xCommon .* sin(ordX .* alpha);
        end        
%         function me = loadobj(me)
%         end
        function [xyLen, pixDot] = get_xyLen(dotSizePix, apRPix)
            % Consider dot's size
            pixDot  = (0:max(round(dotSizePix)-1, 0)) - floor(dotSizePix/2);
            
            % Length
            xyLen   = ...
                round(apRPix(1))*2+1 + abs(pixDot(1))+abs(pixDot(end)) + 2; % Give enough space
            
            % enforce odd number
            xyLen = floor(xyLen / 2) * 2 + 1; 
        end        
        function [xy, c] = xyct2cell(xyct, RDKCol)
            % XYCT2CELL  Convert (nDot x nFr) x (x,y,c,fr) matrix into
            % 1 x nFr cells of (x,y) x nDot and (col2) x nDot.
            %
            % [xy, c] = xyct2cell(xyct, [RDKCol])
            %
            % If RDKCol is provided, replaces v_.xyPix and v_.col2.
            
            t = xyct(:,4);
            T = max(t);
            
            xy = cell(1, T);
            c  = cell(1, T);
            
            for ii = 1:T
                incl = t == ii;
                
                xy{ii} = xyct(incl, 1:2)';
                c{ii}  = xyct(incl, 3)';
            end
            
            if nargin >= 2
                if ~isempty(xy) && ~isempty(xy{1})
                    RDKCol.nDot = size(xy{1}, 2);
                end
                RDKCol.v_.xyPix = xy;
                RDKCol.v_.col2  = c;
                RDKCol.n_.xyPix = length(xy);
                RDKCol.n_.col2  = length(c);
            end
        end
	end
end