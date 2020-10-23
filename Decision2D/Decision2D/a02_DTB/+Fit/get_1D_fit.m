function [th, L, file] = get_1D_fit(i_subj, dim, parad, varargin)
files0.RT = {
    '../Data_2D/Fit.D1.Bounded.Main/sbj=DX+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=C+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=0.mat'
    '../Data_2D/Fit.D1.Bounded.Main/sbj=DX+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=C+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=0.mat'
    '../Data_2D/Fit.D1.Bounded.Main/sbj=MA+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=C+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=0.mat'
    '../Data_2D/Fit.D1.Bounded.Main/sbj=MA+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=C+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=0.mat'
    '../Data_2D/Fit.D1.Bounded.Main/sbj=VL+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=C+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=0.mat'
    '../Data_2D/Fit.D1.Bounded.Main/sbj=VL+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=C+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=0.mat'
% 
%     '../Data_2D/Fit.D1.Bounded.Main/sbj=DX+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat'
%     '../Data_2D/Fit.D1.Bounded.Main/sbj=DX+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat'
%     '../Data_2D/Fit.D1.Bounded.Main/sbj=MA+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat'
%     '../Data_2D/Fit.D1.Bounded.Main/sbj=MA+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat'
%     '../Data_2D/Fit.D1.Bounded.Main/sbj=VL+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat'
%     '../Data_2D/Fit.D1.Bounded.Main/sbj=VL+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=M+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat'
    };
files0.sh = {
    };
files = files0.(parad);
n_dim = 2;
n_subj = numel(files) / 2;
files = reshape(files, [n_dim, n_subj])';

if ischar(i_subj)
    i_subj = find(strcmp(i_subj(1:2), Data.Consts.subjs_RT));
end

file = files{i_subj, dim};
L = load(file);
th = L.res.th;
end