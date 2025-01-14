%Initialization
clc; clear all;
syms q1 q2 q3 q4 q5 q6 
syms len1 len2 len3 len4 

l1 = 112; l2 = 265; l3= 210; l4= 130; lx=50; % link lengths
% Goal Point Coordinates
xg = 200;
yg = 0;
zg = 40;

% Goal Orientation in radians
yaw = 0 * (pi / 180);    % Rotation around Z-axis
pitch = 180 * (pi / 180); % Rotation around Y-axis
roll = 0 * (pi / 180);   % Rotation around X-axis

% Rotation matrix around the X-axis (Roll)
Rx = [1 0 0;
      0 cos(roll) -sin(roll);
      0 sin(roll) cos(roll)];

% Rotation matrix around the Y-axis (Pitch)
Ry = [cos(pitch) 0 sin(pitch);
      0 1 0;
      -sin(pitch) 0 cos(pitch)];

% Rotation matrix around the Z-axis (Yaw)
Rz = [cos(yaw) -sin(yaw) 0;
      sin(yaw) cos(yaw) 0;
      0 0 1];

% Combined rotation matrix R60 (Yaw * Pitch * Roll)
R60 = Rz * Ry * Rx;

% wrist position

Pg = [xg;yg;zg];
Pw = Pg - l4*(R60(:,3))

T60 = [R60 Pg;0 0 0 1];
%% Forward Kinematics
DH = [0 -lx 0 0;
      0 0 len1 q1;
      -pi/2 0 0 q2-pi/2;
      0 len2 0 pi/2+q3;
      pi/2 0 len3 q4;
      -pi/2 0 0 q5;
      pi/2 0 len4 q6];

for i=1:7
    T(:,:,i) = [cos(DH(i,4)) -sin(DH(i,4)) 0 DH(i,2);
            sin(DH(i,4))*cos(DH(i,1)) cos(DH(i,4))*cos(DH(i,1)) -sin(DH(i,1)) -sin(DH(i,1))*DH(i,3);
            sin(DH(i,4))*sin(DH(i,1)) cos(DH(i,4))*sin(DH(i,1)) cos(DH(i,1)) cos(DH(i,1))*DH(i,3);
            0 0 0 1];
end
 
FK_full = T(:,:,1)*T(:,:,2)*T(:,:,3)*T(:,:,4)*T(:,:,5)*T(:,:,6)*T(:,:,7);
FK = T(:,:,1)*T(:,:,2)*T(:,:,3)*T(:,:,4)*T(:,:,5);
X = simplify(FK(1,4));
Y = simplify(FK(2,4));
Z = simplify(FK(3,4));

%% Inverse Kinematics

% solving for theta1
theta1 = atan2(Pw(2),(Pw(1)+lx));

%solving for theta 3 by applying the cosine law

C3 = ((Pw(1)+lx)^2 + Pw(2)^2 + (Pw(3)-len1)^2 -len2^2 -len3^2)/(2*len2*len3); % cosine of theta3
S3 = sqrt(1-C3^2); % Sine of theta3
theta3 = atan2(S3,C3);
theta3 = double(subs(theta3,{len1,len2,len3},{l1,l2,l3}));
% From squaring x,y,z, terms and solving for theta2
%%
A  = (Pw(1)+lx)^2 + Pw(2)^2 + Pw(3)^2 - len1^2 -len2^2 - len3^2 -2*len2*len3*C3;
alpha = 2*len1*len3*C3;
beta = 2*len1*len3*S3;
gamma = 2*len1*len2;
a2 = (alpha+gamma)^2 + beta^2;
b2 = -2*A*(alpha+gamma);
c2 = A^2 - beta^2;

C2 =(-b2+ sqrt(b2^2-4*(a2*c2)))/(2*a2); % cosine of theta 2 roots([a2 b2 c2])
S2 = sqrt(1-C2(1)^2);  % Sine of theta2
theta2 = atan2(S2,C2);
theta2 = double(subs(theta2,{len1,len2,len3},{l1,l2,l3}));
%%
% computing inverse orientation
% computing T63 and then comparing values
T30 = T(:,:,1)*T(:,:,2)*T(:,:,3)*T(:,:,4);
T30 = double(subs(T30,{q1,q2,q3,len1,len2,len3},{theta1,theta2,theta3,l1,l2,l3}));
R30 = T30(1:3,1:3);
T63 = T(:,:,5)*T(:,:,6)*T(:,:,7);
R63 = T63(1:3,1:3);
R63_val = R30'*R60;

% calculation for theta5
C5 = -R63_val(2, 3); % Cosine of theta5
S5 = sqrt(R63_val(3, 3)^2 + R63_val(1, 3)^2); % Sine of theta5
theta5 = atan2(S5, C5);

% Verify the values and recompute if necessary
if abs(C5) < 1e-6 % Check for near-zero cosine value
    C5 = 0; % Correct any floating point issues
end
% computing theta 6 by comparing values in T63
S6 = -R63_val(2,2)/S5;
C6 = R63_val(2,1)/S5;
theta6 = atan2(S6,C6);

% computing theta4 by comparing values in T63
S4 = R63_val(3,3)/S5;
C4 = R63_val(1,3)/S5;
theta4 = atan2(S4,C4);

IK = [theta1;theta2;theta3;theta4;theta5;theta6];
IK_deg = IK*(180/pi)

%% FK calculations
 
FK_val = double(subs(FK,{q1,q2,q3,q4,q5,q6,len1,len2,len3,len4},{theta1,theta2,theta3,theta4,theta5,theta6,l1,l2,l3,l4}));
FK_full_val = double(subs(FK_full,{q1,q2,q3,q4,q5,q6,len1,len2,len3,len4},{theta1,theta2,theta3,theta4,theta5,theta6,l1,l2,l3,l4}));
X_val = FK_val(1,4);
Y_val = FK_val(2,4);
Z_val = FK_val(3,4);
Xg_val = FK_full_val(1,4);
Yg_val = FK_full_val(2,4);
Zg_val = FK_full_val(3,4);
C_orient = FK_full_val(1:3,1:3);

disp(['Desried Goal Position:   X = ', num2str(xg), ', Y = ', num2str(yg), ', Z = ', num2str(zg)]);
disp(['Computed Goal Position:  X = ', num2str(Xg_val), ', Y = ', num2str(Yg_val), ', Z = ', num2str(Zg_val)]);
disp(['Desired Wrist Position:  X = ', num2str(Pw(1)), ', Y = ', num2str(Pw(2)), ', Z = ', num2str(Pw(3))]);
disp(['Computed Wrist Position: X = ', num2str(X_val), ', Y = ', num2str(Y_val), ', Z = ', num2str(Z_val)]);
disp('Desired Orientation: ');
disp(R60);
disp('Computed Orientation: ');
disp(C_orient);

