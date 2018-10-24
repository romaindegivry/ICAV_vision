%% Test Camera FOV

clc
clear
% close all

%% Camera Properties

% Resolution, pixel
reso = [1280, 720];

% Focal Length, mm
f = 3.5;

% Horizontal FOV, deg
FOV_h = 65;

% Horizontal Sensor Size
d_h = tand(FOV_h / 2) * 2 * f;

% Vertical Sensor Size
d_v = d_h / reso(1) * reso(2);

% Verticl FOV
FOV_v = 2 * atand(d_v / (2 * f));


%% Normal Inclindation

% Altitude, m
alti = 30;

% Inclindation from Vertical, deg
% From Ground Vector, anti-clockwise +
incline = 45;

% Field of View
sweep_angle = [incline - FOV_v / 2 , incline + FOV_v /2];

% X - Displacement
x = alti .* tand(sweep_angle);

%% Display FOV

img = imread('old_images/target_A.png');

figure;
grid on
hold on
surface(0:2, 0:2, zeros(3), img, ...
    'FaceColor','texturemap','EdgeColor','none');

view(0,90 - incline);
campos([-x(1),1.5,30]);
camva(FOV_v);

[ax, el] = view;

%% 

% cameraMatrix