clear; clc; close all;

Nx=10; Ny=20; Nz=30;
Sx=rand(Nx,Ny,Nz);
Sy=rand(Nx,Ny,Nz);
Sz=rand(Nx,Ny,Nz);
delta=[1 2 3];

opts.SolverToleranceRel=1e-4;
opts.SolverToleranceAbs=1e-4;
opts.SolverDevice='GPU';
opts.Verbose=1;
opts.Kernel='cell-centered';
%opts.Kernel='face-crossing';

[P, CGS]=OSMODI(single(Sx),single(Sy),single(Sz),single(delta),opts);

sum(P(:)-single(Sz(:)))


