% © 2024. Syracuse University.
% © 2024. Triad National Security, LLC. All rights reserved.
% This program was produced under U.S. Government contract
% 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
% operated by Triad National Security, LLC for the U.S. Department of
% Energy/National Nuclear Security Administration. All rights in the
% program are reserved by Triad National Security, LLC, and the U.S.
% Department of Energy/National Nuclear Security Administration. The
% Government is granted for itself and others acting on its behalf a
% nonexclusive, paid-up, irrevocable worldwide license in this material
% to reproduce, prepare. derivative works, distribute copies to the
% public, perform publicly and display publicly, and to permit
% others to do so.
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License along
% with this program. If not, see <https://www.gnu.org/licenses/>.

%This code provides a sample data set from the flow over a fully-separated slanted-back cylinder
%aligned with the flow at Re=25,000. It calculates the AVERAGE pressure using our OSMODI solver 
%by applying a Reynolds-Averaged Navier-Stokes formulation.

%Here we show the pre-processing steps to obtain the average pressure gradient
%terms from the Navier Stokes momentum equations in their RANS formulation.

%***Note:**** We use only 30 snapshots for this analysis to minimize the size of the shared file.
%The results converge better with more snapshots. Consider using a large
%number of snapshots for your actual analysis.

clear; clc; close all;
load('NTRPIV_LoftedCyl_Sample.mat');

%% Finds the means and Reynolds stresses
Umean=mean(U,3); Vmean=mean(V,3);
[dUdxMean, dUdyMean]=gradient(pagetranspose(Umean),dx,dy); %Gradient uses meshgrid but our data is in NDgrid
dUdxMean=pagetranspose(dUdxMean); dUdyMean=pagetranspose(dUdyMean); %need to undo pagetranspose

[dVdxMean, dVdyMean]=gradient(pagetranspose(Vmean),dx,dy); %Gradient uses meshgrid but our data is in NDgrid
dVdxMean=pagetranspose(dVdxMean); dVdyMean=pagetranspose(dVdyMean); %need to undo pagetranspose

uPrime=U-Umean;
vPrime=V-Vmean;

%Reynolds stresses
uPrime_uPrime=mean(uPrime.^2,3);
uPrime_vPrime=mean(uPrime.*vPrime,3);
vPrime_vPrime=mean(vPrime.^2,3);

%Gradients of Reynolds Stresses
[dUUdx, dUUdy]=gradient(pagetranspose(uPrime_uPrime),dx,dy); %Gradient uses meshgrid but our data is in NDgrid
dUUdx=pagetranspose(dUUdx); dUUdy=pagetranspose(dUUdy); %need to undo pagetranspose

[dUVdx, dUVdy]=gradient(pagetranspose(uPrime_vPrime),dx,dy); %Gradient uses meshgrid but our data is in NDgrid
dUVdx=pagetranspose(dUVdx); dUVdy=pagetranspose(dUVdy); %need to undo pagetranspose

[dVVdx, dVVdy]=gradient(pagetranspose(vPrime_vPrime),dx,dy); %Gradient uses meshgrid but our data is in NDgrid
dVVdx=pagetranspose(dVVdx); dVVdy=pagetranspose(dVVdy); %need to undo pagetranspose


%% Calculates Source terms from time-resolved Navier-Stokes equations:
rho=1.2; %kg/m3
dPdx=-rho*(Umean.*dUdxMean +Vmean.*dUdyMean + dUUdx + dUVdy); 
dPdy=-rho*(Umean.*dVdxMean +Vmean.*dVdyMean + dUVdx + dVVdy); 

%Makes sure all the NaNs are matching
nanMask=isnan(dPdx) | isnan(dPdy);
dPdx(nanMask)=nan; dPdy(nanMask)=nan; %Both fields have to have matching nan masks

%% Calculates pressure from here
opts.SolverToleranceRel=1e-4;
opts.SolverToleranceAbs=1e-4;
opts.SolverDevice='GPU';
opts.Kernel='cell-centered';
opts.Verbose=1;

delta=[dx dy];

xinf=365; yinf=233;
Uinf=sqrt(mean(U(xinf,yinf,:)).^2+mean(V(xinf,yinf,:)).^2);
qinf=0.5*1.2*Uinf.^2;

tic
[Pmean, CGS]=OSMODI(dPdx,dPdy, ones(size(dPdy)),delta,opts);
Pmean=Pmean-Pmean(xinf,yinf); %Reference location has approx. free stream velocity (cp=0)
toc

meanCp=Pmean/qinf;

figure;
imagesc(meanCp'); colormap jet; caxis([-0.5 0]); set(gca,'ydir','normal');
cb=colorbar;
ylabel(cb,'Cp (P/q_\infty)');






