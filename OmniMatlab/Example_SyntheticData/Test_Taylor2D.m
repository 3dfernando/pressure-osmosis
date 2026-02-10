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

clear; clc; close all;

%Generates a Taylor vortex according to Charonko 2010 DOI 10.1088/0957-0233/21/10/105401

t=.2; %Time, s
dt=0.001; %dt for time derivative
H=1e-6; %Vortex strength, m^2
nu=1e-6; %Kinematic Viscosity, m^2/s
rho=1000; %Density, kg/m^3

L0=sqrt(H); %Characteristic Length, m
U0=nu/sqrt(H); %Characteristic Velocity, m/s
T0=L0/U0; %Characteristic Time, s
P0=rho*U0^2; %Characteristic Pressure, Pa

NptsX=357; NptsY=400; r=NptsY/NptsX;

DomainSize=2;
x=linspace(-DomainSize*L0,DomainSize*L0,NptsX); y=linspace(-DomainSize*L0,DomainSize*L0,NptsY); dx=x(2)-x(1); dy=y(2)-y(1);
[X,Y]=ndgrid(x,y);
R=sqrt(X.^2+Y.^2);
Theta=atan2(Y,X);

UR=zeros(size(X));
UTheta=(H/(8*pi)).*(R/(nu*t^2)).*exp(-(R.^2)./(4*nu*t));
UTheta2=(H/(8*pi)).*(R/(nu*(t+dt)^2)).*exp(-(R.^2)./(4*nu*(t+dt)));

P=-(rho*H^2)/(64*(pi^2)*nu*t^3).*exp(-(R.^2)./(2*nu*t));
NormPmax = ((H^2)/(64*(pi^2)*nu*t^3))/(U0^2);

Ux=UR.*cos(Theta) - UTheta.*sin(Theta); Ux2=UR.*cos(Theta) - UTheta2.*sin(Theta);
Uy=UR.*sin(Theta) + UTheta.*cos(Theta); Uy2=UR.*sin(Theta) + UTheta2.*cos(Theta);
Uz=zeros(size(Ux));

dudt=zeros(size(Ux)); dvdt=zeros(size(Ux)); dwdt=zeros(size(Ux)); 
[DUDX, DUDY]=gradient(Ux',dx, dy); DUDX=DUDX'; DUDY=DUDY';
[DVDX, DVDY]=gradient(Uy',dx, dy);  DVDX=DVDX'; DVDY=DVDY';

Sx=(-rho*(dudt+Ux.*DUDX+Uy.*DUDY))/(P0/L0);
Sy=(-rho*(dvdt+Ux.*DVDX+Uy.*DVDY))/(P0/L0);
Sz=zeros(size(Sx));

opts.SolverToleranceRel=1e-4;
opts.SolverToleranceAbs=1e-6;
opts.SolverDevice='GPU';
opts.Verbose=1;
%opts.Kernel='face-crossing';
opts.Kernel='cell-centered';

%Sx(10:20,20:40)=nan;

[P_OSMODI, CGS]=OSMODI(Sx,Sy,Sz,[dx dy]/L0,opts);

disp(['Time Taken for just ' opts.SolverDevice ' computation: ' num2str(CGS(end,3)-CGS(1,3)) 's'])
%%

P_OSMODI=P_OSMODI-P_OSMODI(2,2); %Removes corner value 

figure('color','w');
subplot(1,2,1)
imagesc(P_OSMODI'); colorbar; title('Solver'); daspect([dy dx 1])
subplot(1,2,2)
imagesc(P'/(rho*U0^2)); colorbar; title('Truth'); daspect([dy dx 1])

Ptruth=P/(rho*U0^2);
Err=(P_OSMODI' - Ptruth')./NormPmax;
figure('color','w'); imagesc(Err); colorbar; title(['Error (' opts.SolverDevice ')']); daspect([dy dx 1])

ErrorPercent=100*sqrt(sum(Err(:).^2)./numel(Err));
disp(['Error = ' num2str(ErrorPercent, '%0.3f') '%'])

% figure('color','w'); semilogy(CGS(:,1), CGS(:,2))
% title('CG solver convergence')

