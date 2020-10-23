function file = get_file_BoundedEn(subj, dim_rel)

%%
pth = '../Docs/Data_for_paper/En/DtbEn/Fit.D1.BoundedEn.Main';
files = {
    'sbj=S1+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=1.mat'
    'sbj=S1+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=1.mat'
    'sbj=S2+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=1.mat'
    'sbj=S2+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=1.mat'
    'sbj=S3+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=1.mat'
    'sbj=S3+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=1.mat'
    };
files = reshape(files, [2, 3])';
file = fullfile(pth, files{subj, dim_rel});