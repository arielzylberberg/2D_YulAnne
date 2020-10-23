classdef SubjOld2New_EXCLUDE
    properties (Constant)
        SUBJS = varargin2S({
            % --- Old participant IDs
            'RT', {'S1', 'S2', 'S3'}
            'sh', {'S1', 'S2', 'S3'}
            'eye', {'S1', 'S2', 'S3'}
            'VD', {'ID2', 'ID3'}
            'MANUAL', {'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID9', 'ID16', 'ID18'}
            'unimanual', {'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID9', 'ID16', 'ID18'}
            'bimanual', {'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID9', 'ID16', 'ID18'}
            'unibimanual', {'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID9', 'ID16', 'ID18'}
            %
            % --- New participant IDs
            'S_RT', {'S1', 'S2', 'S3'}
            'S_sh', {'S1', 'S2', 'S3'}
            'S_VD', {'S4', 'S5'}
            'S_eye', {'S1', 'S2', 'S3', 'S4', 'S5'}
            'S_MANUAL', {'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'}
            'S_unimanual', {'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'}
            'S_bimanual', {'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'}
        });
    end
    methods (Static)
        function subj1 = subj_parad2new(subj, parad)
            SUBJS = SubjOld2New_EXCLUDE.SUBJS;
            
            i_subj = find(strcmp(SUBJS.(parad), subj));
            subj1 = SUBJS.(['S_', parad]){i_subj};
        end
    end
end