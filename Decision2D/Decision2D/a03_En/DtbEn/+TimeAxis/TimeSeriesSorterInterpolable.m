classdef TimeSeriesSorterInterpolable < TimeAxis.TimeInheritable
properties % (Access = private)
    y_src = {};
    Time_src % Contains dt_src
    resample_y_src = true % resample according to dt_dst
    
    y_interpolant = {}; % n_tr x 1
    interpolant_method = 'linear';
    interpolant_extrapolation_method = 'none';    
end
% properties
%     to_truncate_first_sec = 0;
%     to_truncate_last_sec = 0;
% end
% % Caching - not implemented yet
% properties (Access = private, Transient)
%     t_dst_prev
%     y_dst_prev
% end
methods
    function Ts = TimeSeriesSorterInterpolable(Time_dst, y_src, dt_src, ...
            varargin)
        
        varargin2props(Ts, varargin);        
        Ts.add_deep_copy({'Time_src'});
        
        if exist('Time_dst', 'var') && ~isempty(Time_dst)
            Ts.set_Time(Time_dst);
        end
        Ts.set_Time_src;
        
        if exist('dt_src', 'var') && ~isempty(dt_src)
            Ts.set_dt_src(dt_src);
        else
            Ts.set_dt_src(1/75); % Default
        end        
        
        if exist('y_src', 'var') && ~isempty(y_src)        
            dt_src = Ts.Time_src.dt;
            dt_dst = Ts.dt;
            if Ts.resample_y_src && dt_dst > dt_src
                assert(abs(dt_dst - round(dt_dst / dt_src) * dt_src) ...
                    < 1e-3, ...
                    'Only integer multiple is supported!');
                C = varargin2C({
                    'truncate_st', 0
                    'truncate_en', 0
                    'n_bin_to_pool', round(dt_dst / dt_src)
                    });
                y_src = pool_time(y_src, C{:});
                Ts.Time_src.dt = dt_dst;
            end            
            Ts.set_y_src(y_src);
        end
    end
end
%% Interface - y_dst
methods
    function [c, S] = get_ts_cell(Ts, varargin)
        % [c, S] = get_ts_cell(Ts, varargin)
        %
        % 't_dst', []
        % 'rows', []
        % 't0', []
        % 'truncate_first_sec', 0
        % 'truncate_last_sec', 0
        S = varargin2S(varargin, {
            't_dst', []
            'to_flip_time', false
            'rows', []
            't0', []
            'truncate_first_sec', -inf
            'truncate_last_sec', -inf
            });
        
        if isempty(S.t_dst) || (ischar(S.t_dst) && isequal(S.t_dst, ':'))
            S.t_dst = Ts.get_t_dst;
        else
            assert(isvector(S.t_dst) && isnumeric(S.t_dst));
        end
        
        if isempty(S.rows) || (ischar(S.rows) && isequal(S.rows, ':'))
            S.rows = 1:Ts.get_n_tr;
        else
            assert(isvector(S.rows) && isnumeric(S.rows));
        end
        n_row = numel(S.rows);
        
        if isempty(S.t0)
            S.t0 = zeros(n_row, 1);
        else
            assert(isvector(S.t0) && isnumeric(S.t0));
            assert(length(S.t0) == n_row);
        end
        
        c = cell(n_row, 1);
        dur_src = Ts.get_dur_src;
        dur_src = dur_src(S.rows);
        dt = Ts.dt;
        
        for i_row = 1:n_row
            t0 = S.t0(i_row);
            dur_src1 = dur_src(i_row);
            
            if isempty(Ts.y_src{i_row}) % y_interp)
                c1 = [];
            else
                if S.to_flip_time
                    t_dst = t0 - S.t_dst;
                else
                    t_dst = S.t_dst - t0;
                end

                % Truncate first and last
                t_incl = (t_dst >= S.truncate_first_sec) ...
                       & (t_dst <= dur_src1 - S.truncate_last_sec);
                t_dst = t_dst(t_incl);

                %%
                is_done = false;
                if Ts.Time_src.dt == dt
                    ix_dst = t_dst / dt;
                    if all(abs(ix_dst - round(ix_dst)) ./ dt < 0.01)
                        ix_dst = round(ix_dst);
                        ix_src = 0:(length(Ts.y_src{i_row}) - 1);
%                         ix_src = round(y_interp.GridVectors{1} / dt);

                        ix_dst1 = min(max(ix_dst, ix_src(1)), ix_src(end));
                        c1 = Ts.y_src{i_row}(ix_dst1 + 1);
%                         c1 = y_interp.Values(ix_dst1 + 1);
                        c1((ix_dst < ix_src(1)) | (ix_dst > ix_src(end))) = nan;
                        
                        is_done = true;
                    end
                end
                
                if ~is_done
                    if isempty(Ts.y_interpolant{i_row})
                        c1 = [];
                    else
                        c1 = Ts.y_interpolant{i_row}(t_dst);
                    end
                end
            end       
            c{i_row} = c1;
            
%             c{i_row} = Ts.get_ts_cell_unit( ...
%                 Ts.y_interpolant{i_row}, ...
%                 S, S.t0(i_row), dur_src(i_row));
        end
        
%         parfor i_row = 1:n_row
%             if isempty(Ts.y_interpolant{i_row})
%                 c{i_row} = [];
%             else
%                 if S.to_flip_time
%                     t_dst = S.t0(i_row) - S.t_dst;
%                 else
%                     t_dst = S.t_dst - S.t0(i_row);
%                 end
% 
%                 % Truncate first and last
%                 t_incl = (t_dst >= S.truncate_first_sec) ...
%                        & (t_dst <= dur_src(i_row) - S.truncate_last_sec);
%                 t_dst = t_dst(t_incl);
% 
%                 c{i_row} = Ts.y_interpolant{i_row}(t_dst);
%             end          
            
%             row = S.rows(i_row);
%             
%             if isempty(Ts.y_interpolant{i_row})
%                 c{row} = [];
%             else
%                 if S.to_flip_time
%                     t_dst = S.t0(i_row) - S.t_dst;
%                 else
%                     t_dst = S.t_dst - S.t0(i_row);
%                 end
% 
%                 % Truncate first and last
%                 t_incl = (t_dst >= S.truncate_first_sec) ...
%                        & (t_dst <= dur_src(row) - S.truncate_last_sec);
%                 t_dst = t_dst(t_incl);
% 
%                 c{row} = Ts.y_interpolant{i_row}(t_dst);
%             end
%         end
        c = c(S.rows);
    end
    function c = get_ts_cell_unit(Ts, y_interp, S, t0, dur_src)
        if isempty(y_interp)
            c = [];
        else
            if S.to_flip_time
                t_dst = t0 - S.t_dst;
            else
                t_dst = S.t_dst - t0;
            end

            % Truncate first and last
            t_incl = (t_dst >= S.truncate_first_sec) ...
                   & (t_dst <= dur_src - S.truncate_last_sec);
            t_dst = t_dst(t_incl);

            %%
            dt = Ts.dt;
            if Ts.Time_src.dt == dt
                ix_dst = t_dst / dt;
                if all(abs(ix_dst - round(ix_dst)) ./ dt < 0.01)
                    ix_dst = round(ix_dst);
                    ix_src = round(y_interp.GridVectors{1} / dt);
                    
                    ix_dst1 = min(max(ix_dst, ix_src(1)), ix_src(end));
                    c = y_interp.Values(ix_dst1 + 1);
                    c((ix_dst < ix_src(1)) | (ix_dst > ix_src(end))) = nan;
                else
                    c = y_interp(t_dst);
                end
            else
                c = y_interp(t_dst);
            end
        end
    end
    function m = get_ts_mat(Ts, varargin)
        % m = get_ts_mat(Ts, varargin)
        %
        % 't_dst', []
        % 'rows', []
        % 't0', []
        % 'truncate_first_sec', 0
        % 'truncate_last_sec', 0
        C = varargin2C(varargin);
        m = bml.matrix.cell2mat2(Ts.get_ts_cell(C{:}), ...
            'min_width', Ts.nt);
    end
    %% t_dst
    function t_dst = get_t_dst(Ts)
        t_dst = Ts.Time.get_t;
    end
    %% Interface - t0_end
    function t0 = get_t0_end(Ts)
        t = Ts.get_t_src;
        t0 = vVec(t(Ts.get_len_src));
    end
end
%% Interface - y_src
methods
    function set_y_src(Ts, y_src)
        Ts.y_src = row2cell2(y_src);
        Ts.set_nt_src;
        Ts.set_y_interpolant;
    end
    function y_src = get_y_src(Ts)
        y_src = Ts.y_src;
    end
    function n_tr = get_n_tr(Ts)
        n_tr = numel(Ts.y_src);
    end
    function len_src = get_len_src(Ts)
        len_src = cellfun(@length, Ts.y_src);
    end
    %% dt_src
    function set_dt_src(Ts, dt_src)
        Ts.Time_src.set_dt(dt_src);
        Ts.set_nt_src;
        Ts.set_y_interpolant;
    end
    function dt_src = get_dt_src(Ts)
        dt_src = Ts.Time_src.get_dt;
    end
    function set_nt_src(Ts)
        Ts.Time_src.set_nt(Ts.get_max_len_src);
    end
    function max_len_src = get_max_len_src(Ts)
        max_len_src = max(Ts.get_len_src);
        if isempty(max_len_src)
            max_len_src = 0; % FIXIT: should be 0 but results in error.
        end
    end
    %% dur_src
    function v = get_dur_src(Ts)
        v = cellfun(@length, Ts.get_y_src);
        v = v * Ts.Time_src.get_dt;
    end    
    %% t_src
    function t_src = get_t_src(Ts)
        t_src = Ts.Time_src.get_t;
    end
    %% Time_src
    function set_Time_src(Ts, Time_src)
        if exist('Time_src', 'var')
            assert(isa(Time_src, 'TimeAxis.TimeRegularPositive'));
        else
            Time_src = TimeAxis.TimeRegularPositive;
        end
        Ts.Time_src = Time_src;
    end
    function Time_src = get_Time_src(Ts)
        Time_src = Ts.Time_src;
    end
end
%% Internal - Custom subsampling
methods (Hidden)
    function y_dst = get_y(Ts, y_src, t_dst, dt_dst)
        nt = numel(y_src);
        t_src = Ts.Time_src.t(1:nt);
        dt_src = Ts.Time.dt;
        
        
        %%
    end
end
%% Internal - Interpolant
methods (Hidden)
    function set_y_interpolant(Ts, y_interpolant)
        % Set or (if not given) reconstruct interpolant based on y_src and Time_src
        
        n_tr = Ts.get_n_tr;
        
        if exist('y_interpolant', 'var') && ~isempty(y_interpolant)
            assert(iscell(y_interpolant));
            assert(isequal(size(y_interpolant), [n_tr, 1]));
            assert(all(cellfun(@(c) isa(c, 'griddedInterpolant'), y_interpolant)));
            
            Ts.y_interpolant = y_interpolant;
        else
            y_src = Ts.get_y_src;
            y_interpolant = cell(n_tr, 1);
            t_src_all = Ts.get_t_src;
            len_src = Ts.get_len_src;

            for i_tr = 1:n_tr
                t_src = t_src_all(1:len_src(i_tr));
                if length(t_src) > 1
                    y_interpolant{i_tr} = griddedInterpolant( ...
                        t_src, y_src{i_tr}, ...
                        Ts.get_interpolant_method, ...
                        Ts.get_interpolant_extrapolation_method);
                else
                    y_interpolant{i_tr} = [];
                end
            end
            Ts.y_interpolant = y_interpolant;
        end
    end
    function set_interpolant_method(Ts, interpolant_method)
        assert(ischar(interpolant_method));
        prev_interpolant_method = Ts.interpolant_method;
        Ts.interpolant_method = interpolant_method;
        
        if ~strcmp(prev_interpolant_method, interpolant_method) ...
                && ~isempty(Ts.y_interpolant)
            % Update interpolant property
            for ii = 1:Ts.get_n_tr
                Ts.y_interpolant{i_tr}.Method = interpolant_method;
            end
        end
    end
    function interpolant_method = get_interpolant_method(Ts)
        interpolant_method = Ts.interpolant_method;
    end
    function set_interpolant_extrapolation_method(Ts, extrapolation_method)
        assert(ischar(extrapolation_method));
        prev = Ts.interpolant_extrapolation_method;
        Ts.interpolant_extrapolation_method = extrapolation_method;
        
        if ~strcmp(prev, extrapolation_method) ...
                && ~isempty(Ts.y_interpolant)
            % Update interpolant property
            for ii = 1:Ts.get_n_tr
                Ts.y_interpolant{i_tr}.ExtrapolationMethod = extrapolation_method;
            end
        end
    end
    function interpolant_extrapolation_method = get_interpolant_extrapolation_method(Ts)
        interpolant_extrapolation_method = Ts.interpolant_extrapolation_method;
    end
end
%% Demo
methods (Static)
    function Ts = demo_upsample
        %% Toy example
        y_src = {
            [5, 6]
            [10, 12, 13]
            };
        len = cellfun(@length, y_src) - 1;
        dt_src = 1;
        
        Time_dst = TimeAxis.TimeRegularPositive('dt', 0.5, 'max_t', 3);

        Ts = TimeAxis.TimeSeriesSorterInterpolable(Time_dst, y_src, dt_src);
        
        ts_st = Ts.get_ts_mat;
        disp(ts_st);
        assert(bml.matrix.isequal_within_nan( ...
            ts_st, [5 5.5 6 nan nan nan nan; 10 11 12 12.5 13 nan nan]));
        
        ts_en = Ts.get_ts_mat('t0', len, 'to_flip_time', true);
        disp(ts_en);
        assert(bml.matrix.isequal_within_nan( ...
            ts_en, [6 5.5 5 nan nan nan nan; 13 12.5 12 11 10 nan nan]));
    end
    function Ts = demo_downsample
        %%
        y_src = {
            [5, 6]
            [10, 12, 13, 14, 15]
            };
        len = cellfun(@length, y_src) - 1;
        dt_src = 0.5;

        Time_dst = TimeAxis.TimeRegularPositive('dt', 1, 'max_t', 3);
        
        Ts = TimeAxis.TimeSeriesSorterInterpolable(Time_dst, y_src, dt_src);
        
        ts_st = Ts.get_ts_mat;
        disp(ts_st);
%         assert(bml.matrix.isequal_within_nan( ...
%             ts_st, [5 5.5 6 nan nan nan nan; 10 11 12 12.5 13 nan nan]));
%         
        ts_en = Ts.get_ts_mat('t0', len, 'to_flip_time', true);
        disp(ts_en);
%         assert(bml.matrix.isequal_within_nan( ...
%             ts_en, [6 5.5 5 nan nan nan nan; 13 12.5 12 11 10 nan nan]));
    end
end
end