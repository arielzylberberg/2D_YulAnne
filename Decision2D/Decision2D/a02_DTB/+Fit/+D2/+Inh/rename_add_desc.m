% Cannot rename. File name too long.

%% Serial
files = bml.str.Serializer.ls( ...
    'Data_2D/Fit.D2.Inh.MainBatch/*.*', ...
    'allof', {
        'd1=0'
        'd2=0'
        's1=0'
        's2=0'
        'd1f=1'
        'd2f=1'
        's1f=1'
        's2f=1'
        });

bml.file.strrep_filename('cv=0+desc=ser', 'cv=0', 'files', files);
bml.file.strrep_filename('cv=1+desc=ser', 'cv=1', 'files', files);

%% Parallel
files = bml.str.Serializer.ls( ...
    'Data_2D/Fit.D2.Inh.MainBatch/*.*', ...
    'allof', {
        'd1=100'
        'd2=100'
        's1=100'
        's2=100'
        'd1f=1'
        'd2f=1'
        's1f=1'
        's2f=1'
        });

bml.file.strrep_filename('cv=0+desc=par', 'cv=0', 'files', files);
bml.file.strrep_filename('cv=1+desc=par', 'cv=1', 'files', files);

