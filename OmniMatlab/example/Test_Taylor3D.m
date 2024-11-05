clear; clc; close all;

%Generates a Taylor vortex according to Charonko 2010 DOI 10.1088/0957-0233/21/10/105401 3D version

t=0.1; %Time, s
H=1e-6; %Vortex strength, m^2
nu=1e-6; %Kinematic Viscosity, m^2/s
rho=1000; %Density, kg/m^3

L0=sqrt(H); %Characteristic Length, m
U0=nu/sqrt(H); %Characteristic Velocity, m/s
T0=L0/U0; %Characteristic Time, s
P0=rho*U0^2; %Characteristic Pressure, Pa

NptsX=201; NptsY=NptsX; NptsZ=NptsX;

x=linspace(-2*L0,2*L0,NptsX); y=linspace(-2*L0,2*L0,NptsY); z=linspace(-2*L0,2*L0,NptsZ);
dx=x(2)-x(1); dy=y(2)-y(1); dz=z(2)-z(1);

[X,Y,Z]=ndgrid(x,y,z);
R=sqrt(X.^2+Y.^2);
Theta=atan2(Y,X);

UR=zeros(size(X));
UTheta=(H/(8*pi)).*(R/(nu*t^2)).*exp(-(R.^2)./(4*nu*t));

P=-(rho*H^2)/(64*(pi^2)*nu*t^3).*exp(-(R.^2)./(2*nu*t));
NormPmax = ((H^2)/(64*(pi^2)*nu*t^3))/(U0^2);

Ux=UR.*cos(Theta) - UTheta.*sin(Theta);
Uy=UR.*sin(Theta) + UTheta.*cos(Theta);
Uz=zeros(size(Ux));

dudt=zeros(size(Ux)); dvdt=zeros(size(Ux)); dwdt=zeros(size(Ux)); 
[DUDX, DUDY, DUDZ]=gradient(pagetranspose(Ux),dx,dy,dz); DUDX=pagetranspose(DUDX); DUDY=pagetranspose(DUDY); DUDZ=pagetranspose(DUDZ);
[DVDX, DVDY, DVDZ]=gradient(pagetranspose(Uy),dx,dy,dz);  DVDX=pagetranspose(DVDX); DVDY=pagetranspose(DVDY); DVDZ=pagetranspose(DVDZ);

Sx=(-rho*(dudt+Ux.*DUDX+Uy.*DUDY))/(P0/L0);
Sy=(-rho*(dvdt+Ux.*DVDX+Uy.*DVDY))/(P0/L0);
Sz=zeros(size(Sx));

opts.SolverToleranceRel=1e-4;
opts.SolverToleranceAbs=1e-6;
opts.SolverDevice='GPU';
%opts.Kernel='face-crossing';
opts.Kernel='cell-centered';
opts.Verbose=1;

M=zeros(size(Sx));
M(26:40,26:40,26:40)=1; %Makes sure the degeneracies are handled correctly
M=M & rand(size(Sx))>0.5;
Sx(M)=nan;

disp('Starting OSMODI...')
tic
%[P_OSMODI, CGS]=OSMODI(single(Sx),single(Sy),single(Sz),single([dx dy dz]/L0),opts);
[P_OSMODI, CGS]=OSMODI(Sx,Sy,Sz,[dx dy dz]/L0,opts);
toc

%figure; imagesc(squeeze(P_OSMODI(:,ceil(NptsX/2),:)));colorbar

disp(['Time Taken for just GPU computation: ' num2str(CGS(end,3)-CGS(1,3)) 's'])


%% Compares to the solution by the solver

P_OSMODI=P_OSMODI-P_OSMODI(2,2); %Removes corner value 

figure;
subplot(1,2,1)
imagesc(P_OSMODI(:,:,round(NptsZ/2))); colorbar; title('Solver'); daspect([1 1 1])
subplot(1,2,2)
imagesc(P(:,:,round(NptsZ/2))/(rho*U0^2)); colorbar; title('Truth'); daspect([1 1 1])

figure;
subplot(1,2,1)
imagesc(squeeze(P_OSMODI(:,round(NptsY/2),:))); colorbar; title('Solver'); daspect([1 1 1])
subplot(1,2,2)
imagesc(squeeze(P(:,round(NptsY/2),:)/(rho*U0^2))); colorbar; title('Truth'); daspect([1 1 1])


Ptruth=P/(rho*U0^2);
Err=(P_OSMODI - Ptruth)./NormPmax;
ErrorPercent=100*sqrt(sum(Err(:).^2)./numel(Err));
disp(['Error = ' num2str(ErrorPercent, '%0.3f') '%'])

% figure('color','w','Position', [1.3538e+03 530.6000 560 420]); semilogy(CGS(:,1), CGS(:,2))
% title('CG solver convergence')

% pause(2)

clear;