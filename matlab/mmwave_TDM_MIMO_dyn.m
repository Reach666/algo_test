% 警告: MATLAB 先前因底层图形错误而崩溃。为避免在此会话期间再次崩溃，MATLAB 将使用软件 OpenGL
% 而不再使用图形硬件。要保存该设置以供此后的会话使用，请使用 opengl('save', 'software') 命
% 令。有关详细信息，请参见解决底层的图形问题。 
% > In matlab.graphics.internal.initialize (line 15) 
%%
close all;
clear;
%clc;
%% 演示选项
% MultiTarget=0; FindMax=1; DynPlot=0; PlotComplex=0;%单目标 卡尔曼滤波效果
% MultiTarget=0; FindMax=1; DynPlot=1; PlotComplex=0; %单目标 动态过程+滤波效果
% MultiTarget=0; FindMax=0; DynPlot=1; PlotComplex=1;%两个目标 动态过程
% MultiTarget=0; FindMax=0; DynPlot=1; PlotComplex=2;%两个目标 动态过程 图多 慢
% MultiTarget=0; FindMax=0; DynPlot=1; PlotComplex=3;%两个目标 动态过程 图多 很慢
MultiTarget=1; FindMax=0; DynPlot=1; PlotComplex=2;%五个目标 动态过程 图多 慢

%去下文调参数
noise_power=0;
Interpolation=1;%插值
EKF=1;%卡尔曼滤波是否扩展
SpeedRevise=1;
RangeRevise=0;%RD热图未修正  第二修正项意义不大，代表物体的T_frame_valid位移
AngleRevise=1;
TDM_DopplerCalib=1;

%% 雷达系统参数
ComplexSignal=1;
fmin=23.5e9;%24GHz
fmax=27.5e9;
% fmin=24e9;%24GHz
% fmax=24.25e9;
Bmax=fmax-fmin;
f_center_full=(fmin+fmax)/2;
c=3e8;%light velocity
lambda=c/fmin;%or f_center???
T_ramp=0.5e-3;
T_idle=0.01e-3;%0.5e-3; 0.2e-3
Tc=T_ramp+T_idle;%chirp_period=1ms;
slopemax=100e12;%100MHz/us
slope=Bmax/T_ramp; assert(slope<=slopemax)
fs=512e3;
Ts=1/fs;
rangeN=256; assert(T_ramp*fs>=rangeN)%256
B_valid=Ts*rangeN*slope;% about?
f_center_valid=fmin+B_valid/2;
chirpN=32;%32
T_frame_valid=Tc*chirpN;
T_frame_idle=0;
T_frame=T_frame_valid+T_frame_idle;

TxN=2;   chirpN_TDM=chirpN/TxN; assert((chirpN_TDM-round(chirpN_TDM))==0);
RxN=4;
Rx_d=lambda/2; % >lambda/2  aliasing
Tx_coordinate=[0 0 0; 1*RxN*Rx_d 0 0; 2*RxN*Rx_d 0 0; 3*RxN*Rx_d 0 0;]; Tx_coordinate=Tx_coordinate(1:TxN,:);
Rx_coordinate=[lambda 0 0; lambda+Rx_d 0 0; lambda+2*Rx_d 0 0; lambda+3*Rx_d 0 0; lambda+4*Rx_d 0 0; lambda+5*Rx_d 0 0; lambda+6*Rx_d 0 0; lambda+7*Rx_d 0 0;]; Rx_coordinate=Rx_coordinate(1:RxN,:);
%Rx_coordinate=[lambda 0 0; lambda+lambda/2 0 0; lambda 0 lambda/2; lambda+lambda/2 0 lambda/2;];

noise_power=-55;%-50dbm
Pt=10000;


SpeedRevise=1;
RangeRevise=0;%RD热图未修正  第二修正项意义不大，代表物体的T_frame_valid位移
AngleRevise=1;
delta_range=c/2/B_valid;%距离分辨率
max_range=rangeN*delta_range*(ComplexSignal+1)/2;%距离量程(复信号)
delta_speed=c/2/fmin/T_frame_valid;%速度分辨率 注意下文负号
if SpeedRevise
    delta_speed=c/(2*fmin-slope*Ts*rangeN)/T_frame_valid;%速度分辨率修正!!!  c/2/(fmin-B_valid/2)/T_frame_valid
end
max_speed=chirpN_TDM/2*delta_speed;%速度量程(正负)
%Rx_d=lambda/2;
delta_angle=c/fmin/RxN/Rx_d;
if AngleRevise
    delta_angle=c/(fmin-slope*Ts*rangeN/2)/RxN/Rx_d;% fmin-B_valid/2
end

%% FFT参数
range_win = hanning(rangeN);%rectwin(rangeN);%hamming(rangeN);  %加窗
rangeFFTN_all = 1*rangeN;
rangeFFT_ZP_R=rangeFFTN_all/rangeN;%Zero Padding ratio
if ComplexSignal==1
    rangeFFTN=rangeFFTN_all;
else
    rangeFFTN=rangeFFTN_all/2;
end

doppler_win = hanning(chirpN_TDM);%rectwin(chirpN);%hamming(chirpN);
dopplerFFTN = 2*chirpN_TDM;
dopplerFFT_ZP_R=dopplerFFTN/chirpN_TDM;%Zero Padding ratio

angle_win = kaiser(RxN*TxN,2);%rectwin(RxN*TxN);%hamming(RxN*TxN);
angleFFTN = 2*RxN*TxN;%max(64,4*RxN*TxN);
angleFFT_ZP_R=angleFFTN/RxN;%Zero Padding ratio

range_doppler_win=range_win*doppler_win';
%range_doppler_win=repmat( reshape(range_doppler_win,rangeN,[],1) ,[1,1,RxN]);
%range_doppler_angle_win=reshape( reshape(range_doppler_win,[],1)*angle_win' ,rangeN,chirpN_TDM,[]);
range_doppler_angle_win=range_doppler_win.*reshape(angle_win,1,1,[]);

%% 目标状态
frameN=70;

if MultiTarget==1
    targetN=9; % basic*1 near_target*2 x_speed*1 overspeed*1 RCS_change*2 crossbar*2
    target_coord=[0 2 0; 0.2 0.5 0.2; 0.22 0.71 0.21;  3  3 -0.5; -3 1 0; -1   1 0;  1 1.01 0; 2 2 0; 2.01   2 0;]; target_coord=target_coord(1:targetN,:);
    target_speed=[0 1 0; 0.5 0.5 0.5; 0.51 0.53 0.55;  -6 0    0;  0 5 0;  2 1.5 0; -2 1.5  0; 0 2 0;    0 1.99 0]; target_speed=target_speed(1:targetN,:);
    target_state=[target_coord target_speed]';
    Tf=T_frame/100;
    state_trans_mat=[1 0 0 Tf 0 0; 0 1 0 0 Tf 0; 0 0 1 0 0 Tf; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]; sim_precision=T_frame/Tf;
    target_rcs=[0.01 0.001 0.001 0.1 1 0.01 0.01 0.1 0.1]; target_rcs=target_rcs(1:targetN);
elseif MultiTarget==0 %两个目标
    if FindMax
        targetN=1;
    else
        targetN=2;
    end
    t_c_Y=2;
    target_coord=[0 t_c_Y+0.5 0;0 t_c_Y 0]; target_coord=target_coord(1:targetN,:);%[0 t_c_Y+0.5 0;0 t_c_Y 0]
    target_speed=[2 0 0;-2 -2 0]; target_speed=target_speed(1:targetN,:);%[2 0 0;-2 -2 0]
    target_acc=[0 0 0;0 0 0]; target_acc=target_acc(1:targetN,:);
    target_state=[target_coord target_speed target_acc ones(targetN,1)]';
    stateN=length(target_state);
    Tf=T_frame/100;%/1
    sim_precision=T_frame/Tf;
    k=10;
    state_trans_mat=[1 0 0 Tf 0 0 Tf^2/2 0 0 0; 0 1 0 0 Tf 0 0 Tf^2/2 0 0; 0 0 1 0 0 Tf 0 0 Tf^2/2 0; ...
        0 0 0 1 0 0 Tf 0 0 0; 0 0 0 0 1 0 0 Tf 0 0; 0 0 0 0 0 1 0 0 Tf 0;...
        -k 0 0 0 0 0 0 0 0 0; 0 -k 0 0 0 0 0 0 0 t_c_Y*k; 0 0 -k 0 0 0 0 0 0 0;...
        0 0 0 0 0 0 0 0 0 1];
    target_rcs=[0.05 0.05]; target_rcs=target_rcs(1:targetN);
end

%% 卡尔曼初始化
EKF=1;
dt=T_frame;
xN_=9;xN=10;
%%系统状态矩阵
Fk_=[1 0 0 dt 0 0 dt^2/2 0 0; 0 1 0 0 dt 0 0 dt^2/2 0; 0 0 1 0 0 dt 0 0 dt^2/2; ...
        0 0 0 1 0 0 dt 0 0; 0 0 0 0 1 0 0 dt 0; 0 0 0 0 0 1 0 0 dt;...
        -0 0 0 0 0 0 1 0 0; 0 -0 0 0 0 0 0 1 0; 0 0 -0 0 0 0 0 0 1;];
if EKF==0
    fade=1.0;
elseif EKF==1
    fade=1.0;
    %fade=1.1;%1
end
%%过程噪声
qk=0.0001;
Qk=qk*[dt^4/4 0 0  dt^3/2 0 0  dt^2/2 0 0 0; 0 dt^4/4 0 0 dt^3/2 0 0 dt^2/2 0 0; 0 0 dt^4/4 0 0 dt^3/2 0 0 dt^2/2 0; ...
    dt^3/2 0 0 dt^2 0 0 dt 0 0 0; 0 dt^3/2 0 0 dt^2 0 0 dt 0 0;  0 0 dt^3/2 0 0 dt^2 0 0 dt 0; ...
    dt^2/2 0 0 dt 0 0 1 0 0 0; 0 dt^2/2 0 0 dt 0 0 1 0 0; 0 0 dt^2/2 0 0 dt 0 0 1 0; ...
    0 0 0 0 0 0 0 0 0 0];
% qk=0.01;
% Qk=qk*[dt^5/20 0 0  dt^4/8 0 0  dt^3/6 0 0 0; 0 dt^5/20 0 0 dt^4/8 0 0 dt^3/6 0 0; 0 0 dt^5/20 0 0 dt^4/8 0 0 dt^3/6 0; ...
%     dt^4/8 0 0 dt^3/3 0 0 dt^2/2 0 0 0; 0 dt^4/8 0 0 dt^3/3 0 0 dt^2/2 0 0;  0 0 dt^4/8 0 0 dt^3/3 0 0 dt^2/2 0; ...
%     dt^3/6 0 0 dt^2/2 0 0 dt 0 0 0; 0 dt^3/6 0 0 dt^2/2 0 0 dt 0 0; 0 0 dt^3/6 0 0 dt^2/2 0 0 dt 0; ...
%     0 0 0 0 0 0 0 0 0 0];
if EKF==0
    Qk_=1*Qk(1:9,1:9); 
    qk_a_ratio=30000; Qk_(7,7)=qk_a_ratio*Qk_(7,7); Qk_(8,8)=qk_a_ratio*Qk_(8,8); Qk_(9,9)=qk_a_ratio*Qk_(9,9);
elseif EKF==1
    Qk_=1*Qk(1:9,1:9); 
    qk_a_ratio=30000; Qk_(7,7)=qk_a_ratio*Qk_(7,7); Qk_(8,8)=qk_a_ratio*Qk_(8,8); Qk_(9,9)=qk_a_ratio*Qk_(9,9);
end
%%量测输出矩阵
if EKF==0
    Hk=[1 zeros(1,8);zeros(1,1) 1 zeros(1,7);zeros(1,2) 1 zeros(1,6)];
    Hk_=1*Hk;
elseif EKF==1
    Hk_=[zeros(1,9);zeros(1,9);zeros(1,9);zeros(1,9)];
end
%%量测噪声 和信噪比实时有关
if EKF==0
    record_rk=0.0004;
    Rk_=record_rk*eye(3); Rk_(2,2)=Rk_(2,2)*0.05;Rk_(3,3)=Rk_(3,3)*1e-9;
elseif EKF==1
    record_rk=0.0001;
    Rk_=record_rk*eye(4); Rk_(2,2)=Rk_(2,2)*0.5;Rk_(3,3)=Rk_(3,3)*1e-9;Rk_(4,4)=Rk_(4,4)*5; % Rk_(4,4)速度项会影响准确度
end
%%初始估计
xk_post=zeros(xN_,1); xk_post(2)=1;
I=eye(xN_, xN_);
Pk_post=1e3*I;%1e-1*I;

sensorN=3+EKF;
record_xk_post=zeros(9,frameN);
record_yk=zeros(sensorN,frameN);
record_yk_pr=zeros(sensorN,frameN);
record_yk_post=zeros(sensorN,frameN);
record_Pk_post=zeros(1,frameN);

%% 卡尔曼滤波效果绘图初始化
%DynPlot=0;
%figure('ButtonDownFcn',@(src,event) pause(5));
if FindMax
    record_target_state=zeros(stateN,frameN);
end
record_detected_x=zeros(frameN,1);
record_detected_y=zeros(frameN,1);
record_detected_z=zeros(frameN,1);
record_detected_R=zeros(frameN,1);
record_detected_v=zeros(frameN,1);
record_detected_sinA=zeros(frameN,1);

%% 状态更新 系统执行
for frame_n=1:frameN
    if frame_n>1
        if MultiTarget==0
            wk=mvnrnd(zeros(xN,1),Qk)';
            target_state=state_trans_mat^sim_precision*target_state + wk;
        else
            target_state=state_trans_mat^sim_precision*target_state;
        end
        %target_state=state_trans_mat^sim_precision*target_state + [zeros(6,targetN);wgn(2,targetN,10);zeros(2,targetN)];%10dB
    end
    target_state_T=target_state';
    target_coord=target_state_T(:,1:3);
    target_speed=target_state_T(:,4:6);

    %% 回波信号生成
    %认为一个chirp中 速度不变
    
    %矩形贴片天线
    antenna_beta=2*pi/lambda; 
    antenna_W=lambda/4; 
    antenna_L=lambda/4;
    A_Rx=antenna_W*antenna_L;
    
    %Tx天线方向图增益
    Tx_target_Cartesian=target_coord(:,:)-reshape(Tx_coordinate',1,[],TxN);
    Tx_target_Cartesian_y=reshape(Tx_target_Cartesian(:,1,:),targetN,TxN);
    Tx_target_Cartesian_z=reshape(Tx_target_Cartesian(:,2,:),targetN,TxN);
    Tx_target_Cartesian_x=reshape(Tx_target_Cartesian(:,3,:),targetN,TxN);
    [Tx_target_spherical_az,Tx_target_spherical_el,Tx_target_spherical_r]=cart2sph(Tx_target_Cartesian_x,Tx_target_Cartesian_y,Tx_target_Cartesian_z); %[az,el,r] = cart2sph(x,y,z)
    Tx_target_spherical_el=pi/2-Tx_target_spherical_el;
    antenna_f=sinc(1/pi*antenna_beta*antenna_W/2.*sin(Tx_target_spherical_el).*sin(Tx_target_spherical_az)).*cos(antenna_beta*antenna_L/2.*sin(Tx_target_spherical_el).*cos(Tx_target_spherical_az));
    Gt=sqrt(   ( cos(Tx_target_spherical_az).*antenna_f ).^2 + ( cos(Tx_target_spherical_el).*sin(Tx_target_spherical_az).*antenna_f ).^2    );
    Gt=permute(reshape(Gt,targetN,TxN,1,1,1),[3 4 5 1 2]);
    %Rx天线方向图增益
    Rx_target_Cartesian=target_coord(:,:)-reshape(Rx_coordinate',1,[],RxN);
    Rx_target_Cartesian_y=reshape(Rx_target_Cartesian(:,1,:),targetN,RxN);
    Rx_target_Cartesian_z=reshape(Rx_target_Cartesian(:,2,:),targetN,RxN);
    Rx_target_Cartesian_x=reshape(Rx_target_Cartesian(:,3,:),targetN,RxN);
    [Rx_target_spherical_az,Rx_target_spherical_el,Rx_target_spherical_r]=cart2sph(Rx_target_Cartesian_x,Rx_target_Cartesian_y,Rx_target_Cartesian_z); %[az,el,r] = cart2sph(x,y,z)
    Rx_target_spherical_el=pi/2-Rx_target_spherical_el;
    antenna_f=sinc(1/pi*antenna_beta*antenna_W/2.*sin(Rx_target_spherical_el).*sin(Rx_target_spherical_az)).*cos(antenna_beta*antenna_L/2.*sin(Rx_target_spherical_el).*cos(Rx_target_spherical_az));
    Gr=sqrt(   ( cos(Rx_target_spherical_az).*antenna_f ).^2 + ( cos(Rx_target_spherical_el).*sin(Rx_target_spherical_az).*antenna_f ).^2    );
    Gr=permute(reshape(Gr,targetN,RxN,1,1,1),[3 4 2 1 5]);
    %传统口径天线
    %Gt=1;Gr=Gt;
    %A_Rx=Gr*lambda^2/(2*pi);  %Pr=Ptar*A_Rx./(4*pi*d_Rx_target.^2);
    

    %frame_data=zeros(rangeN,chirpN,RxN);
    %frame_data_part=zeros(rangeN,chirpN,RxN,targetN,TxN);
    t=(0:Ts:Ts*(rangeN-1))';
    n_chirp_TDM=(0:chirpN_TDM-1);
    d_Rx_target=sum( (reshape(Rx_coordinate',1,[],RxN)-target_coord(:,:)).^2 ,2).^(1/2);%size: target 1(coord) Rx
    v_Rx_target_radial=dot(repmat( reshape(target_speed(:,:),[],3,1) ,1,1,RxN),(target_coord(:,:)-  reshape(Rx_coordinate',1,[],RxN) ),2)./d_Rx_target;%size: target 1(coord) Rx
    d_Tx_target=sum( (reshape(Tx_coordinate',1,[],TxN)-target_coord(:,:)).^2 ,2).^(1/2);%size: target 1(coord) Rx
    v_Tx_target_radial=dot(repmat( reshape(target_speed(:,:),[],3,1) ,1,1,TxN),(target_coord(:,:)-  reshape(Tx_coordinate',1,[],TxN) ),2)./d_Tx_target;%size: target 1(coord) Tx
    d_Rx_target=permute(reshape(d_Rx_target,1,1,targetN,RxN,1),[1 2 4 3 5]);
    v_Rx_target_radial=permute(reshape(v_Rx_target_radial,1,1,targetN,RxN,1),[1 2 4 3 5]);
    d_Tx_target=permute(reshape(d_Tx_target,1,1,1,targetN,TxN),[1 2 3 4 5]);
    v_Tx_target_radial=permute(reshape(v_Tx_target_radial,1,1,1,targetN,TxN),[1 2 3 4 5]);
    RCS=reshape(target_rcs,1,1,1,[],1);
    Ptar=Pt.*Gt.*RCS./(4*pi*d_Tx_target.^2);
    %Pr=Ptar*A_Rx./(4*pi*d_Rx_target.^2); %传统口径天线
    Pr=Ptar.*Gr.*A_Rx./(4*pi*d_Rx_target.^2);
    Ar=sqrt(Pr);
    tao=(d_Tx_target+v_Tx_target_radial.*(t+Tc*n_chirp_TDM*TxN)+d_Rx_target+v_Rx_target_radial.*(t+Tc*n_chirp_TDM*TxN))/c;
    tao=tao + (v_Tx_target_radial.*reshape((Tc*(0:TxN-1)),1,1,1,1,[]) + v_Rx_target_radial.*reshape((Tc*(0:TxN-1)),1,1,1,1,[]))/c;% 加上第五维  chirp_TDM全局修改
    %rangeN,chirpN,RxN,targetN,TxN
    if ComplexSignal
        frame_data_part=Ar.*exp(1i*2*pi*(fmin*tao+slope/2*(2*tao.*t-tao.*tao)));
    else
        frame_data_part=Ar.*cos(1i*2*pi*(fmin*tao+slope/2*(2*tao.*t-tao.*tao)));
    end
    frame_data_part=permute(frame_data_part,[1 2 3 5 4]);
    frame_data=sum(frame_data_part,5);
    noise = reshape( wgn(rangeN*chirpN_TDM,RxN*TxN, noise_power) ,rangeN,chirpN_TDM,RxN,TxN);%-50dbm
    frame_data = frame_data + noise;

%     t=0:Ts:Ts*(rangeN-1);
%     frame_data=zeros(rangeN,chirpN,RxN);
%     for n_chirp=0:chirpN-1
%         for n_Rx=1:RxN
%             for n_target=1:targetN
%                 d_Rx_target=norm(Rx_coordinate(n_Rx,:)-target_coord(n_target,:));
%                 v_Rx_target_radial=target_speed(n_target,:)*(target_coord(n_target,:)-Rx_coordinate(n_Rx,:))'/d_Rx_target;
%                 RCS=target_rcs(n_target);
%                 for n_Tx=1:TxN
%                     d_Tx_target=norm(Tx_coordinate(n_Tx,:)-target_coord(n_target,:));
%                     v_Tx_target_radial=target_speed(n_target,:)*(target_coord(n_target,:)-Tx_coordinate(n_Tx,:))'/d_Rx_target;
%                     Ptar=Pt*Gt*RCS/(4*pi*d_Tx_target^2);
%                     A_Rx=Gr*lambda^2/(2*pi);
%                     Pr=Ptar*A_Rx/(4*pi*d_Rx_target^2);
%                     Ar=sqrt(Pr);
% 
%                     tao=(d_Tx_target+v_Tx_target_radial*(t+Tc*n_chirp)+d_Rx_target+v_Rx_target_radial*(t+Tc*n_chirp))/c;
%                     if ComplexSignal==1
%                         frame_data(:,n_chirp+1,n_Rx)=frame_data(:,n_chirp+1,n_Rx) ...
%                             + reshape( Ar*exp(1i*2*pi*(-fmin*tao+slope/2*(2*tao.*t-tao.*tao))) ,[],1,1);%after mixer
%     %近似                    
%     %                     frame_data(n_Rx,n_chirp+1,:)=frame_data(n_Rx,n_chirp+1,:) ...
%     %                        + reshape( Ar*exp(1i*2*pi*((2*slope*(d_Rx_target+v_Rx_target_radial*Tc*n_chirp)/c-2*fmin*v_Rx_target_radial/c)*t...
%     %                        -2*fmin*(d_Rx_target+v_Rx_target_radial*Tc*n_chirp)/c)) ,1,1,[]);%after mixer
%     %去掉交叉项
%     %                     frame_data(n_Rx,n_chirp+1,:)=frame_data(n_Rx,n_chirp+1,:) ...
%     %                        + reshape( Ar*exp(1i*2*pi*((2*slope*(d_Rx_target)/c-2*fmin*v_Rx_target_radial/c)*t...
%     %                        -2*fmin*(d_Rx_target+v_Rx_target_radial*Tc*n_chirp)/c)) ,1,1,[]);%after mixer
%     %观察交叉项
%     %                     frame_data(n_Rx,n_chirp+1,:)=frame_data(n_Rx,n_chirp+1,:) ...
%     %                        + reshape( Ar*exp(1i*2*pi*((2*slope*(d_Rx_target/10000+2*v_Rx_target_radial*Tc*n_chirp)/c)*t...
%     %                        -2*fmin*(d_Rx_target+v_Rx_target_radial*Tc*n_chirp)/c/10000)) ,1,1,[]);%after mixer
%                     else
%                         frame_data(:,n_chirp+1,n_Rx)=frame_data(:,n_chirp+1,n_Rx) ...
%                             + reshape( Ar*cos(2*pi*(-fmin*tao+slope/2*(2*tao.*t-tao.*tao))) ,[],1,1);%after mixer
%                     end
%                 end
%             end  
%         end
%     end
%     noise = reshape( wgn(rangeN*chirpN,RxN, noise_power) ,rangeN,chirpN,RxN);%-50dbm
%     frame_data = frame_data + noise;

    %% TDM MIMO 数据绘图演示
    if frame_n==1
        n_target=1;
        calib=2;
        if calib==1
            frame_data_part(:,:,:,2,n_target)=frame_data_part(:,:,:,2,n_target).*exp(1i*(pi*22/64+1*pi)); % 22是rdmap的doppler维度数字频率
        elseif calib==2
            v_temp=v_Rx_target_radial(:,:,1,n_target);
            frame_data_part(:,:,:,2,n_target)=frame_data_part(:,:,:,2,n_target).*exp(1i*2*v_temp*Tc/lambda*2*pi*1);
            if TxN>=3
                frame_data_part(:,:,:,3,n_target)=frame_data_part(:,:,:,3,n_target).*exp(1i*2*v_temp*Tc/lambda*2*pi*2);
            end
            if TxN>=4
                frame_data_part(:,:,:,4,n_target)=frame_data_part(:,:,:,4,n_target).*exp(1i*2*v_temp*Tc/lambda*2*pi*3);
            end
        end
        frame_data_part_reshape=reshape( frame_data_part(:,:,:,:,n_target),rangeN,chirpN_TDM,[]);
        plot(reshape(real(frame_data_part_reshape(:,1,:)),rangeN,[]))

        figure;
        plot(reshape(real(frame_data_part_reshape(1,1,:)),1,[]));hold on;

        plot(reshape(real(frame_data_part_reshape(1:10,1,1:RxN)),10,[]),'r');hold on;
        plot(reshape(real(frame_data_part_reshape(1:10,1,RxN+1:2*RxN)),10,[]),'g--')
        plot(reshape(real(frame_data_part_reshape(1:10,2,RxN+1:2*RxN)),10,[]),'b--')
        if TxN>=3
            plot(reshape(real(frame_data_part_reshape(1:10,1,2*RxN+1:3*RxN)),10,[]),'r--');
        end
        if TxN>=4
            plot(reshape(real(frame_data_part_reshape(1:10,1,3*RxN+1:4*RxN)),10,[]),'g--')
        end

        figure;
        subplot(1,2,1);
        plot(abs( fftshift(fft(reshape(frame_data_part_reshape(1,1,:),1,[]),8*2*RxN)) )); %v=4 v=0.6
        subplot(1,2,2);
        plot(abs( fftshift(fft(reshape(frame_data_part_reshape(1,1,:),1,[]).*angle_win',8*2*RxN)) )); %v=4 v=0.6
        
%         SIZE = get(0);					% 获取显示屏的像素尺寸
%         h = figure();					% 创建图形窗口
%         set(h, 'position', get(0,'ScreenSize'));	% 设置图形窗口位置和外尺寸为屏幕大小
%         pause(1);
    end
    
    %% 数据绘图
%     if frame_n==1
%         mesh(real((squeeze(frame_data(:,:,1)))));xlabel('time');ylabel('chirp');
%     end

    %% FFT2+TDM(calib)+FFT
    frame_data_win=frame_data.*range_doppler_win;
    RDfft=fftshift(fft2(frame_data_win,rangeFFTN_all,dopplerFFTN),2);
    if ComplexSignal==0
        RDfft=RDfft(1:rangeFFTN,:,:);
    end
    
%     RDmap_amp2=abs(Rx1_Rx2_RDfft);%已经平方了
%     RDmap_amp=sqrt(RDmap_amp2);
    RDmap_amp=mean(abs(RDfft),[3 4]); %sum
    RDmap_amp2=RDmap_amp.^2;
    
    TDM_DopplerCalib=1;
    if TDM_DopplerCalib
        doppler_phase_calib=2*pi*( -(1:dopplerFFTN)+dopplerFFTN/2+1 )/TxN/dopplerFFTN;
        doppler_phase_calib=doppler_phase_calib'.*(0:TxN-1);
        RDfft=RDfft.*reshape(exp(1i*doppler_phase_calib),1,dopplerFFTN,1,TxN);
    end
    RDfft=reshape(RDfft,rangeFFTN,dopplerFFTN,RxN*TxN);
    % 可以先用2D-CFAR 以减少FFT计算量
    
    range_fft_sum=squeeze( sum(abs(RDfft).^2,[2 3]) );%功率
    RDAfft=fftshift(fft(RDfft.*reshape(angle_win,1,1,[]),angleFFTN,3),3);
    RDAfft_amp=abs(RDAfft);
    %RAfft(n_range,:)=RAfft(n_range,:)+abs(temp_fft);
    RAfft=squeeze(sum(RDAfft_amp,2));
    

    %% FFTn
%     frame_data_win=frame_data.*range_doppler_angle_win;
%     RDAfft = fftshift(fftshift( fftn(frame_data_win,[rangeFFTN_all dopplerFFTN angleFFTN]) ,2),3);
%     if ComplexSignal==0
%         RDAfft=RDAfft(1:rangeFFTN,:,:);
%     end
%     RAfft = zeros(rangeFFTN,angleFFTN);
%     RAfft(:,:)=RAfft(:,:)+squeeze( sum(abs(RDAfft),2) );
%     range_fft_sum=zeros(1,rangeFFTN);
%     range_fft_sum(:)=range_fft_sum(:)+ squeeze( sum(abs(RDAfft).^2,[2 3]) );%功率

    %% my 3维FFT处理
%     FFT3D=0;
%     if FFT3D==1
%         %距离FFT
%         range_profile = zeros(rangeFFTN,chirpN,RxN);
%         range_fft_sum = zeros(1,rangeFFTN);
%         for n_Rx=1:RxN
%            for n_chirp=1:chirpN
%               temp=(frame_data(:,n_chirp,n_Rx)).*range_win;    %加窗函数
%               %temp=(data_radar(:,m,k)-data_radar(:,m+1,k)).*range_win;    %脉冲对消 加窗函数
%               temp_fft=fft(temp,rangeFFTN_all);    %对每个chirp做N点FFT
%               range_profile(:,n_chirp,n_Rx)=temp_fft(1:rangeFFTN);
%               range_fft_sum(:)=range_fft_sum(:)+abs(temp_fft(1:rangeFFTN)).^2;%功率
%            end
%         end
% 
%         %多普勒FFT
%         RDfft = zeros(rangeFFTN,dopplerFFTN,RxN);
%         for n_Rx=1:RxN
%             for n_range=1:rangeFFTN
%               temp=range_profile(n_range,:,n_Rx).*(doppler_win)';    
%               temp_fft=fftshift(fft(temp,dopplerFFTN));    %对rangeFFT结果进行M点FFT
%               %temp_fft=[temp_fft(1:dopplerFFTN/2-2) zeros(1,2*3-1-1) temp_fft(dopplerFFTN/2+3:dopplerFFTN)];%高通滤波
%               RDfft(n_range,:,n_Rx)=temp_fft; 
%             end
%         end
%     %     range_profile_=bsxfun(@times,range_profile,reshape(doppler_win,1,[],1));
%     %     RDfft=fftshift(fft(range_profile_,dopplerFFTN,2),2); 
% 
%         %Rx1_Rx2
%         Rx1_rdmap=squeeze(RDfft(:,:,1));
%         Rx2_rdmap=squeeze(RDfft(:,:,2));
%         RDmap_amp2=abs(Rx1_rdmap.*Rx2_rdmap);
% 
%         %角度FFT
%         RDAfft = zeros(rangeFFTN,dopplerFFTN,angleFFTN);
%         RAfft = zeros(rangeFFTN,angleFFTN);
%         for n_range=1:rangeFFTN   %range
%             for n_doppler=1:dopplerFFTN   %chirp
%               temp=reshape(RDfft(n_range,n_doppler,:),1,[]);    
%               temp_fft=fftshift(fft(temp,angleFFTN));    %对2D FFT结果进行Q点FFT
%               RDAfft(n_range,n_doppler,:)=temp_fft;  
%               RAfft(n_range,:)=RAfft(n_range,:)+abs(temp_fft);
%             end
%         end
%     end

    %% 3D CFAR
    if 1
        RDAfft_amp2=RDAfft_amp.^2;
        if FindMax==0
            cfar_threshold=100*mean(RDAfft_amp2(:));
            [range_index,doppler_index,angle_index]=ind2sub( size(RDAfft_amp2), find(RDAfft_amp2>cfar_threshold) );
        else
            RDAfftmax=max(max(RDAfft_amp2));
            [range_index,doppler_index,angle_index]=ind2sub( size(RDAfft_amp2), find(RDAfft_amp2==RDAfftmax) );
        end
        RDAfft_amp2_cfar=zeros(rangeFFTN,dopplerFFTN,angleFFTN);
        cfar_detected_N=length(doppler_index);
        detected_amp2=zeros(1,cfar_detected_N); %或者使用 二次插值+二次函数三点式求幅值
        for detected_n=1:cfar_detected_N
            temp=RDAfft_amp2(range_index(detected_n),doppler_index(detected_n),angle_index(detected_n));
            RDAfft_amp2_cfar(range_index(detected_n),doppler_index(detected_n),angle_index(detected_n))=temp;
            detected_amp2(detected_n)=temp;
        end
    end
    
    
    %% 2D CFAR
%     fft2D_abs2=RDmap_amp2;
%     if FindMax==0
%         cfar_threshold=10*mean(fft2D_abs2(:))*ones(rangeFFTN,dopplerFFTN);
%         [range_index,doppler_index]=find(fft2D_abs2>cfar_threshold);
%     else
%         RDAfftmax=max(max(fft2D_abs2));
%         [range_index,doppler_index]=find(fft2D_abs2==RDAfftmax);
%     end
%     fft2D_abs2_cfar=zeros(rangeFFTN,dopplerFFTN);
%     cfar_detected_N=length(doppler_index);
%     detected_amp2=zeros(1,cfar_detected_N);
%     for detected_n=1:cfar_detected_N
%         temp=fft2D_abs2(range_index(detected_n),doppler_index(detected_n));
%         fft2D_abs2_cfar(range_index(detected_n),doppler_index(detected_n))=temp;
%         detected_amp2(detected_n)=temp;
%     end

    %% 目标计算
    if FindMax==1
        Interpolation=1;
    else
        Interpolation=1;
    end
    
    detected_A=zeros(1,detected_n);
    detected_sinA=zeros(1,detected_n);
    detected_B=zeros(1,detected_n);
    detected_sinB=zeros(1,detected_n);
    detected_R=zeros(1,detected_n);
    detected_x=zeros(1,detected_n);
    detected_y=zeros(1,detected_n);
    detected_z=zeros(1,detected_n);
    detected_v=zeros(1,detected_n);
    for detected_n=1:cfar_detected_N
        %使用1T2R相位差测角
        DeltaPhaseAngle=0;
        if DeltaPhaseAngle==1
            Rx1_RDfft=squeeze(RDfft(:,:,1,1));
            Rx2_RDfft=squeeze(RDfft(:,:,2,1));
            Rx1_Rx2_RDfft=Rx1_RDfft.*conj(Rx2_RDfft);% amp mult  phase minus
        
            % delta_phase=phase( RDfft(range_index(detected_n),doppler_index(detected_n),1) )...
            %     -phase( RDfft(range_index(detected_n),doppler_index(detected_n),2) );
            % if delta_phase>pi
            %     delta_phase=delta_phase-2*pi;
            % elseif delta_phase<-pi
            %     delta_phase=delta_phase+2*pi;
            % end
            delta_phase=phase(Rx1_Rx2_RDfft( range_index(detected_n),doppler_index(detected_n)) );
            sinA=-lambda*delta_phase/(2*pi*Rx_d);
            if AngleRevise
                sinA=-c/(fmin-slope*Ts*rangeN/2)*delta_phase/(2*pi*Rx_d);%c/(fmin-slope*Ts*rangeN/2)/RxN/Rx_d;
            end
        end
        
        if Interpolation
            x1temp=range_index(detected_n); x2temp=doppler_index(detected_n);x3temp=angle_index(detected_n);
            if x3temp>1 && x3temp<angleFFTN
                detected_amp=RDAfft_amp(x1temp,x2temp,x3temp);%y2
                detected_plus_amp=RDAfft_amp(x1temp,x2temp,x3temp+1);%y3
                detected_minus_amp=RDAfft_amp(x1temp,x2temp,x3temp-1);%y1
                if detected_amp>=detected_plus_amp && detected_amp>=detected_minus_amp
                    x3temp_=((detected_amp-detected_minus_amp)*(x3temp+1+x3temp)-(detected_plus_amp-detected_amp)*(x3temp+x3temp-1))...
                        /(detected_plus_amp+detected_minus_amp-2*detected_amp)/(-2);
                    sinA=( x3temp_-1-angleFFTN/2 )/angleFFT_ZP_R *delta_angle;
                else
                    sinA=(angle_index(detected_n)-1-angleFFTN/2)/angleFFT_ZP_R *delta_angle;
                    detected_amp2(detected_n)=detected_amp2(detected_n)/10000;
                end
            else %或者循环插值
                sinA=(angle_index(detected_n)-1-angleFFTN/2)/angleFFT_ZP_R *delta_angle;
            end
        else
            sinA=(angle_index(detected_n)-1-angleFFTN/2)/angleFFT_ZP_R *delta_angle;
        end
        
        if Rx_d<lambda/2 || AngleRevise
            sinA((sinA>1))=1;
            sinA((sinA<-1))=-1;
        end
        detected_sinA(detected_n)=( sinA );
        A=asin( sinA );%A=180/pi* asin( sinA );
        detected_A(detected_n)=A;
        sinB=0;
        detected_sinB(detected_n)=sinB;
        B=asin( sinB );
        detected_B(detected_n)=B;
        if Interpolation
            x1temp=range_index(detected_n); x2temp=doppler_index(detected_n);x3temp=angle_index(detected_n);
            if x1temp>1 && x1temp<rangeFFTN
                detected_amp=RDAfft_amp(x1temp,x2temp,x3temp);%y2
                detected_plus_amp=RDAfft_amp(x1temp+1,x2temp,x3temp);%y3
                detected_minus_amp=RDAfft_amp(x1temp-1,x2temp,x3temp);%y1
                if detected_amp>=detected_plus_amp && detected_amp>=detected_minus_amp
                    x1temp_=((detected_amp-detected_minus_amp)*(x1temp+1+x1temp)-(detected_plus_amp-detected_amp)*(x1temp+x1temp-1))...
                        /(detected_plus_amp+detected_minus_amp-2*detected_amp)/(-2);
                    R=(x1temp_-1)/rangeFFT_ZP_R *delta_range;
                else
                    R=(range_index(detected_n)-1)/rangeFFT_ZP_R *delta_range;
                    detected_amp2(detected_n)=detected_amp2(detected_n)/2;
                end
            else %或者循环插值
                R=(range_index(detected_n)-1)/rangeFFT_ZP_R *delta_range;
            end
        else
            R=(range_index(detected_n)-1)/rangeFFT_ZP_R *delta_range;
        end
        detected_R(detected_n)= R;
        detected_x(detected_n)= R*sinA;
        detected_y(detected_n)= R*sqrt(1-sinA^2-sinB^2);
        detected_z(detected_n)= R*sinB;
        if Interpolation
            x1temp=range_index(detected_n); x2temp=doppler_index(detected_n);x3temp=angle_index(detected_n);
            if x2temp>1 && x2temp<dopplerFFTN
                detected_amp=RDAfft_amp(x1temp,x2temp,x3temp);%y2
                detected_plus_amp=RDAfft_amp(x1temp,x2temp+1,x3temp);%y3
                detected_minus_amp=RDAfft_amp(x1temp,x2temp-1,x3temp);%y1
                if detected_amp>=detected_plus_amp && detected_amp>=detected_minus_amp
                    x2temp_=((detected_amp-detected_minus_amp)*(x2temp+1+x2temp)-(detected_plus_amp-detected_amp)*(x2temp+x2temp-1))...
                        /(detected_plus_amp+detected_minus_amp-2*detected_amp)/(-2);
                    v=( -x2temp_+(dopplerFFTN/2+1) )/dopplerFFT_ZP_R *delta_speed;
                else
                    v=( -doppler_index(detected_n)+(dopplerFFTN/2+1) )/dopplerFFT_ZP_R *delta_speed;
                    detected_amp2(detected_n)=detected_amp2(detected_n)/100;
                end
            else %或者循环插值
                v=( -doppler_index(detected_n)+(dopplerFFTN/2+1) )/dopplerFFT_ZP_R *delta_speed;
            end
        else
            v=( -doppler_index(detected_n)+(dopplerFFTN/2+1) )/dopplerFFT_ZP_R *delta_speed;
        end
        detected_v(detected_n)=v;
        %detected_delta_phase(detected_n)=delta_phase;
    end
    %% 卡尔曼滤波
    if FindMax
        if EKF==0
            xk_pr = Fk_*xk_post;
            Pk_pr = fade * Fk_*Pk_post*Fk_' + Qk_;
            Rk__=Rk_;
            Rk__=Rk_*detected_R^4; %Rk__=Rk_*r_^4;  %!!!!
            Kk = Pk_pr*Hk_'*(Hk_*Pk_pr*Hk_'+Rk__)^-1;
            %Kkconst=Kk;Kk=Kkconst;%稳态卡尔曼系数
            yk = [detected_x;detected_y;detected_z];
            xk_post = xk_pr + Kk*(yk-Hk_*xk_pr);
            Pk_post = (I-Kk*Hk_)*Pk_pr*(I-Kk*Hk_)' + Kk*Rk__*Kk';% =(I-Kk*Hk_)*Pk_pr
            
            record_xk_post(:,frame_n)=xk_post;
            record_yk(:,frame_n)=yk;
            record_yk_pr(:,frame_n)=Hk_*xk_pr;
            record_yk_post(:,frame_n)=Hk_*xk_post;
            record_Pk_post(frame_n)=sum(diag(Pk_post));
        elseif EKF==1%反而效果不好，可能是需要专注于距离估计，而不是速度和加速度估计一起计算代价函数
            xk_pr = Fk_*xk_post;
            Pk_pr = fade * Fk_*Pk_post*Fk_' + Qk_;
            
            x_=xk_pr(1);y_=xk_pr(2);z_=xk_pr(3);vx_=xk_pr(4);vy_=xk_pr(5);vz_=xk_pr(6);
            r_=sqrt(x_^2+y_^2+z_^2)+1e-6; sqrtyz_=sqrt(y_^2+z_^2)+1e-6; sqrtxy_=sqrt(x_^2+y_^2)+1e-6;
            Hk_(1,1)=x_/r_;Hk_(1,2)=y_/r_;Hk_(1,3)=z_/r_;
            Hk_(2,1)=sqrtyz_/r_^2;Hk_(2,2)=-x_*y_/r_^2/sqrtyz_;Hk_(2,3)=-x_*z_/r_^2/sqrtyz_;
            Hk_(3,1)=-x_*z_/r_^2/sqrtxy_;Hk_(3,2)=-y_*z_/r_^2/sqrtxy_;Hk_(3,3)=sqrtxy_/r_^2;
            Hk_(4,1)=(y_*(vx_*y_-vy_*x_)+z_*(vx_*z_-vz_*x_))/r_^3;
            Hk_(4,2)=(x_*(vy_*x_-vx_*y_)+z_*(vy_*z_-vz_*y_))/r_^3;
            Hk_(4,3)=(x_*(vz_*x_-vx_*z_)+y_*(vz_*y_-vy_*z_))/r_^3;
            Hk_(4,4)=x_/r_;Hk_(4,5)=y_/r_;Hk_(4,6)=z_/r_;
            Rk__=Rk_;
            Rk__=Rk_*detected_R^4; %Rk__=Rk_*r_^4;  %!!!!
            %Rk__=Rk_/detected_amp2;
            Kk = Pk_pr*Hk_'*(Hk_*Pk_pr*Hk_'+Rk__)^-1;
            %Kkconst=Kk;Kk=Kkconst;%稳态卡尔曼系数
            yk=[detected_R detected_A detected_B detected_v]';
            yk_pr=[r_ atan(x_/sqrtyz_) atan(z_/sqrtxy_) (x_*vx_+y_*vy_+z_*vz_)/r_]';
            xk_post = xk_pr + Kk*(yk-yk_pr);
            Pk_post = (I-Kk*Hk_)*Pk_pr*(I-Kk*Hk_)' + Kk*Rk__*Kk';% =(I-Kk*Hk_)*Pk_pr
            
            record_xk_post(:,frame_n)=xk_post;
            record_yk(:,frame_n)=yk;
            record_yk_pr(:,frame_n)=yk_pr;
            x_=xk_post(1);y_=xk_post(2);z_=xk_post(3);vx_=xk_post(4);vy_=xk_post(5);vz_=xk_post(6);
            r_=sqrt(x_^2+y_^2+z_^2)+1e-6; sqrtyz_=sqrt(y_^2+z_^2)+1e-6; sqrtxy_=sqrt(x_^2+y_^2)+1e-6;
            yk_post=[r_ atan(x_/sqrtyz_) atan(z_/sqrtxy_) (x_*vx_+y_*vy_+z_*vz_)/r_]';
            record_yk_post(:,frame_n)=yk_post;
            record_Pk_post(frame_n)=sum(diag(Pk_post));
        end
    end
    
    %% 记录轨迹
    if FindMax
        record_target_state(:,frame_n)=target_state;
        record_detected_x(frame_n)=detected_x;
        record_detected_y(frame_n)=detected_y;
%         record_detected_z(frame_n)=detected_z;
%         record_detected_R(frame_n)=detected_R;
%         record_detected_v(frame_n)=detected_v;
%         record_v_Rx_target_radial(frame_n,:)=v_Rx_target_radial;
%         record_detected_sinA(frame_n)=detected_sinA;
%         record_detected_A(frame_n)=detected_A;
%         record_A(frame_n)=atan(target_state(1)/target_state(2));
    end

    %% 绘图
    if DynPlot
        image_n=frame_n;
        if image_n==1
            figure;
            set(gcf,'unit','centimeters','position',[2,1,30,16]);%左下角为原点 横8~(8+20) cm 纵4~(4+10) cm
            %set(imh, 'erasemode', 'none');
            %figure('units','normalized','position',[0.01,0.01,0.8,0.8])
        end
        
        if PlotComplex==0
            subplot_col=2;subplot_row=2;
            RealScatter3=1;    Scatter3Guide=1;  scatter_xlim=4; %1
            Rplot=0;
            RDmesh=0;    RDmeshABS2=0; RDmeshCFAR=0; 
            RDmeshCFARdetect=0;
            RDmap=2; %2
            RAmesh=0;
            XYmesh=0;
            XYscatter=3; XYscatter_hold=0; %3
        elseif PlotComplex==1
            subplot_col=2;subplot_row=2;
            RealScatter3=1;    Scatter3Guide=1;  scatter_xlim=4; %1
            Rplot=0;
            RDmesh=0;    RDmeshABS2=0; RDmeshCFAR=0; 
            RDmeshCFARdetect=0;
            RDmap=2; %2
            RAmesh=0;
            XYmesh=0;
            XYscatter=3; XYscatter_hold=1; %3
        elseif PlotComplex==2
            subplot_col=2;subplot_row=3;
            RealScatter3=1;    Scatter3Guide=1;  scatter_xlim=4;
            Rplot=2;
            RDmesh=3;    RDmeshABS2=0; RDmeshCFAR=0; 
            RDmeshCFARdetect=0;
            RDmap=6; 
            RAmesh=5;
            XYmesh=0; %5
            XYscatter=4; XYscatter_hold=1;
        elseif PlotComplex==3
            subplot_col=2;subplot_row=3;
            RealScatter3=1;    Scatter3Guide=1;  scatter_xlim=4;
            Rplot=2;
            RDmesh=0;    RDmeshABS2=0; RDmeshCFAR=0; 
            RDmeshCFARdetect=3;
            RDmap=6;
            RAmesh=0;
            XYmesh=5;
            XYscatter=4; XYscatter_hold=1;
        end

        % 原始散点
        if RealScatter3
            subplot(subplot_col,subplot_row,RealScatter3);
            if targetN>1
                scatter3(target_coord(:,1),target_coord(:,2),target_coord(:,3),...
                    mapminmax(log(target_rcs(:)'),3,15).^2,sum(target_speed.^2,2),...
                    'MarkerFaceColor',[0 .75 .75])%'MarkerEdgeColor','k',...   target_coord(:,3)
            else
                scatter3(target_coord(:,1),target_coord(:,2),target_coord(:,3),...
                    'MarkerFaceColor',[0 .75 .75])%'MarkerEdgeColor','k',...
            end
            hold on;
            if Scatter3Guide
                for n_target=1:targetN
                    plot3([target_coord(n_target,1) target_coord(n_target,1)],...
                        [target_coord(n_target,2) target_coord(n_target,2)],...
                        [target_coord(n_target,3) -10],'--')
                    %hold on;
                end
            end
            scatter3(Rx_coordinate(:,1),Rx_coordinate(:,2),Rx_coordinate(:,3),'*');%hold on;
            hold off;
            if image_n==1

            end
            axis([-scatter_xlim scatter_xlim 0 max([5;max(target_coord(:,2))]) -1 1]);
            %xlim([-scatter_xlim scatter_xlim]); ylim([0 max([5;max(target_coord(:,2))])]); zlim([-1 1]);
            xlabel('x(m)');ylabel('y(m)');zlabel('z(m)');
            view(10,65);%view(130,20)
        end


        % 距离维
        if Rplot
            subplot(subplot_col,subplot_row,Rplot);
            if image_n==1
                X_Rplot=(0:rangeFFTN-1)/rangeFFT_ZP_R *delta_range;
            end
            plot(X_Rplot,range_fft_sum);
            xlabel('距离(m)');ylabel('功率');
            %plot(X_Rplot,squeeze(abs(range_profile(1,1,:))));%xlabel('距离(m)');ylabel('幅度');
        end


        % RD视图/abs/cfar
        if RDmesh
            subplot(subplot_col,subplot_row,RDmesh);
            if image_n==1
                [X_RDmesh,Y_RDmesh]=meshgrid((dopplerFFTN/2:-1:-dopplerFFTN/2+1)/dopplerFFT_ZP_R *delta_speed,(0:rangeFFTN-1)/rangeFFT_ZP_R *delta_range);
                if RangeRevise
                    for y=1:dopplerFFTN
                        yspeed=X_RDmesh(1,y);
                        Y_RDmesh(:,y)=Y_RDmesh(:,y)+(fmin/slope-T_frame_valid/2)*yspeed;
                    end
                end
            end
            
            if RDmeshABS2==0
                mesh(X_RDmesh,Y_RDmesh,(abs(RDmap_amp))); 
            else
                mesh(X_RDmesh,Y_RDmesh,((RDmap_amp2))); 
                if RDmeshCFAR==1
                    hold on;mesh(X_RDmesh,Y_RDmesh, cfar_threshold); hold off; % CFAR
                end
            end

            ylabel('距离(m)');xlabel('速度(m/s)');zlabel('信号幅值');
            title('RD视图');
            if RDmeshABS2==1
                zlabel('信号功率'); title('RD功率');
            end
        end


        % RD abs2 CFAR detected视图
        if RDmeshCFARdetect
            subplot(subplot_col,subplot_row,RDmeshCFARdetect);
            if image_n==1
                [X_RDmeshCFARdetect,Y_RDmeshCFARdetect]=meshgrid((dopplerFFTN/2:-1:-dopplerFFTN/2+1)/dopplerFFT_ZP_R *delta_speed,(0:rangeFFTN-1)/rangeFFT_ZP_R *delta_range);
                if RangeRevise
                    for y=1:dopplerFFTN
                        yspeed=X_RDmeshCFARdetect(1,y);
                        Y_RDmeshCFARdetect(:,y)=Y_RDmeshCFARdetect(:,y)+(fmin/slope-T_frame_valid/2)*yspeed;
                    end
                end
            end
            
            mesh(X_RDmeshCFARdetect,Y_RDmeshCFARdetect,((RDmap_amp2))); hold on;
            % for n=1:rangeFFTN
            %     plot3(X(n,:),Y(n,:),cfar_threshold(n,:),'r:')
            %     hold on;
            % end
            for n=1:dopplerFFTN
                plot3(X_RDmeshCFARdetect(:,n),Y_RDmeshCFARdetect(:,n),cfar_threshold(:,n),'c-');%hold on;
            end
            for detected_n=1:cfar_detected_N
                plot3(detected_v(detected_n),detected_R(detected_n),detected_amp2(detected_n),'ro');
                    %%%%'MarkerSize',3,'MarkerFaceColor','r');
            end%text(x,y,z,txt)
            hold off;
            ylabel('距离(m)');xlabel('速度(m/s)');zlabel('信号功率');title('RD CFAR');
        end


        % RD热图
        if RDmap
            subplot(subplot_col,subplot_row,RDmap);
            if image_n==1
                X_RDmap=(0:rangeFFTN-1)/rangeFFT_ZP_R *delta_range;
                Y_RDmap=(dopplerFFTN/2:-1:-dopplerFFTN/2+1)/dopplerFFT_ZP_R *delta_speed;
            end
            %imshow(RDmap_amp2/max(max(RDmap_amp2))*10) %normalize
            imagesc(X_RDmap,Y_RDmap,RDmap_amp2');
            xlabel('距离(m)');ylabel('速度(m/s)');title('RD热图');
        end


        % R-角度 图
        if RAmesh
            subplot(subplot_col,subplot_row,RAmesh);
            if image_n==1
                [R_RAmesh,sinA_RAmesh]=meshgrid((0:rangeFFTN-1)/rangeFFT_ZP_R *delta_range, ( (-angleFFTN/2:1:angleFFTN/2-1)/angleFFT_ZP_R *delta_angle) );
                if Rx_d<lambda/2 || AngleRevise
                    sinA_RAmesh((sinA_RAmesh>1))=1;
                    sinA_RAmesh((sinA_RAmesh<-1))=-1;
                end
                A_RAmesh=180/pi*asin(sinA_RAmesh);
            end
            mesh(R_RAmesh,A_RAmesh,RAfft')
            ylim([-90 90]); xlabel('距离(m)'); ylabel('角度');
        end


        % XY三维图
        if XYmesh
            subplot(subplot_col,subplot_row,XYmesh);
            if image_n==1
                [R_XYmesh,sinA_XYmesh]=meshgrid((0:rangeFFTN-1)/rangeFFT_ZP_R *delta_range, ( (-angleFFTN/2:1:angleFFTN/2)/angleFFT_ZP_R *delta_angle ) );%多一个单元
                if Rx_d<lambda/2 || AngleRevise
                    sinA_XYmesh((sinA_XYmesh>1))=1;
                    sinA_XYmesh((sinA_XYmesh<-1))=-1;
                end
                %[X,Y]=pol2cart(A,R);
                X_XYmesh=R_XYmesh.*sinA_XYmesh;
                Y_XYmesh=R_XYmesh.*sqrt(1-sinA_XYmesh.^2);
            end
            RAfft_plus=[RAfft RAfft(:,1)];%多一个单元
            mesh(X_XYmesh,Y_XYmesh,RAfft_plus')
            %imagesc(X_XYmesh,Y_XYmesh,RDmap_amp2);
            xlabel('X(m)'); ylabel('Y(m)');title('XY视图');
            view(10,80);
        end


        % XY散点图
        if XYscatter
            subplot(subplot_col,subplot_row,XYscatter);
            if image_n==1

            end
            detected_v_=detected_v;
            if length(detected_v_)>4
                detected_v_(1)=max_speed*0.7;detected_v_(2)=-max_speed*0.7;
                detected_amp2_=detected_amp2; %detected_amp2_(1)=0.2;detected_amp2_(2)=0.01;
                if targetN>1
                    if XYscatter_hold
                        scatter(detected_x,detected_y,50*log2(detected_amp2/50+1),detected_v_);
                    else
                        scatter(detected_x,detected_y,mapminmax(log(detected_amp2_),1,3).^4,[detected_v_]);
                    end
                else
                    if XYscatter_hold
                        scatter(detected_x,detected_y,50*log2(detected_amp2/50+1),detected_v_);
                    else
                        scatter(detected_x,detected_y,[],detected_v_);%scatter(detected_x,detected_y,detected_amp2,detected_v_);
                    end
                end
            else
                scatter(detected_x,detected_y,[],detected_v_);
            end

            %stem3(detected_x,detected_y,detected_amp2);scatter3(detected_x,detected_y,detected_amp2);
            axis([-scatter_xlim scatter_xlim 0 max([5;max(target_coord(:,2))])]);
            %xlim([-scatter_xlim scatter_xlim]); ylim([0 max([5;max(target_coord(:,2))])]);
            %axis equal;
            xlabel('X(m)'); ylabel('Y(m)');title('XY散点图');
            if XYscatter_hold
                hold on;
            end
        end

        %
        pause(0.0001)
        %f_getframe(:,j)=getframe;
        %movie2avi(mov, 'myPeaks.avi', 'compression', 'None');
    end
    
end
%movie(f_getframe,3,10)

%% 卡尔曼滤波结果
if FindMax
    %状态
    record_xk=record_target_state(1:9,:);
    subplot(2,3,1);plot(record_xk','DisplayName','record_xk');
    ylim([-max(max(record_xk_post(:,frameN/2:frameN))) max(max(record_xk_post(:,frameN/2:frameN)))])
    subplot(2,3,4);plot(record_xk_post','DisplayName','record_xk_post');
    ylim([-max(max(record_xk_post(:,frameN/2:frameN))) max(max(record_xk_post(:,frameN/2:frameN)))])
    %量测
    subplot(2,3,2);plot(record_yk');
    ylim([-max(max(record_yk_post(:,frameN/2:frameN))) max(max(record_yk_post(:,frameN/2:frameN)))])
    hold on;plot(record_yk_post');hold off;
    %subplot(2,3,5);plot(record_yk_post');
    ylim([-max(max(record_yk_post(:,frameN/2:frameN))) max(max(record_yk_post(:,frameN/2:frameN)))])
    %位置信息
    subplot(2,3,3);plot(record_detected_x(:),record_detected_y(:))%frameN/2:frameN
    hold on;plot(record_target_state(1,:),record_target_state(2,:));hold off;
    subplot(2,3,6);plot(record_xk_post(1,:),record_xk_post(2,:));
    hold on;plot(record_target_state(1,:),record_target_state(2,:));hold off;
    %新息协方差验证 一次
    rk=yk-Hk_*xk_pr;%rk新息
    rk_cov=rk*rk';
    rk_cov_expect=Hk_*Pk_pr*Hk_'+Rk_
    %新息协方差验证 均值
    record_rk=record_yk-record_yk_pr;
    subplot(2,3,5);plot(record_rk');ylim([-max(max(record_rk(:,frameN/2:frameN))) max(max(record_rk(:,frameN/2:frameN)))])
    record_rk_=record_rk(:,frameN/2:frameN);
    rk_sum=record_rk_*record_rk_';
    rk_mean=rk_sum/(frameN/2)
    %新息白噪声验证   应该rkri<2*rkrk
    record_rk__=record_rk(:,end-frameN/20:1:end);
    rk_sum_=record_rk__'*record_rk__;
    %mesh(rk_sum_)
    rkri=sum(sum(rk_sum_,1),2);% rk*ri'
    rkrk=sum(diag(rk_sum_));% rk*rk'
    %估计方差
    %subplot(2,3,3);plot(record_Pk_post);ylim([0 2*max(record_Pk_post(frameN/2:frameN))])
end

