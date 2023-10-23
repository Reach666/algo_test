import numpy as np

# 雷达系统参数
ComplexSignal = True
fmin = 23.5e9  # 24GHz
fmax = 27.5e9
Bmax = fmax - fmin
f_center_full = (fmin + fmax) / 2
c = 3e8  # light velocity
_lambda = c / fmin
T_ramp = 409.6e-6 # T_ramp = Ts * rangeN
T_idle = 80e-6
Tc = T_ramp + T_idle
slope = Bmax / T_ramp
fs = 2.5e6 / 4
Ts = 1 / fs
rangeN = 256
B_valid = Ts * rangeN * slope
f_center_valid = fmin + B_valid / 2
chirpN = 128
T_frame_valid = Tc * chirpN
T_frame_idle = 0
T_frame = T_frame_valid + T_frame_idle

# 天线排布
TxN = 4
RxN = 4
Tx_coord = [[0,0,0],[1*RxN,0,0],[2*RxN,0,0],[3*RxN,0,0]]
Rx_coord = [[0,0,0],[1,0,0],[2,0,0],[3,0,0]]
Rx_d = _lambda / 2
Tx_coord = np.array(Tx_coord)[:TxN] * Rx_d
Rx_coord = np.array(Rx_coord)[:RxN] * Rx_d
TxDM_mode = 'DDM' # 'TDM' 'DDM' None
if TxDM_mode == 'TDM':
    TxDM_pattern = np.diag([1]*TxN) # (chirp,TxN)    TDM: np.diag([1]*TxN)
elif TxDM_mode == 'DDM':
    empty_band = 1
    TxDM_pattern = np.arange(chirpN).reshape(-1,1) * (np.arange(TxN) / (TxN + empty_band) * 2 * np.pi)
    TxDM_pattern = np.exp(1j * TxDM_pattern)
else:
    TxDM_pattern = np.ones((1,TxN))
assert chirpN % TxN == 0 and chirpN % len(TxDM_pattern) == 0
chirpN_DM = chirpN // TxN

# 功率参数
noise_power = -55 # -50dbm
Pt = 10000

# 目标状态
targetN = 2
target_coord = np.array([[0, 2.5, 0], [0, 2, 0]]) # (targetN,3)
target_speed = np.array([[1, 0, 0], [-1, -1, 0]]) # (targetN,3)
target_rcs = np.array([0.05, 0.05])

# 天线方向图
Gt = 1
Gr = 1
# 矩形贴片天线
antenna_beta = 2 * np.pi / _lambda
antenna_W= _lambda / 4
antenna_L= _lambda / 4
A_Rx = antenna_W * antenna_L
'''    
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
'''


# 计算雷达信号
t = np.arange(0, Ts * rangeN, Ts).reshape(rangeN,1) # (rangeN,1)
n_chirp = np.arange(0, chirpN).reshape(chirpN,1,1,1) # (chirpN,1,1,1)
chirp_target_coord = target_coord + target_speed * (0 + Tc * n_chirp) # (chirpN,1,targetN,3)     1 for rangeN

coord_Rx_target = chirp_target_coord - Rx_coord.reshape(RxN,1,1,1,3) # (RxN,chirpN,1,targetN,3)
d_Rx_target = np.linalg.norm(coord_Rx_target,axis=-1) # (RxN,chirpN,1,targetN)
coord_Tx_target = chirp_target_coord - Tx_coord.reshape(TxN,1,1,1,1,3) # (TxN,1,chirpN,1,targetN,3)
d_Tx_target = np.linalg.norm(coord_Tx_target,axis=-1) # (TxN,1,chirpN,1,targetN)
tao = (d_Tx_target + d_Rx_target) / c # (TxN,RxN,chirpN,1,targetN)

RCS = target_rcs # (targetN)
Ptar = Pt * Gt * RCS / (4 * np.pi * d_Tx_target ** 2) # (TxN,1,chirpN,1,targetN)
Pr = Ptar * Gr * A_Rx / (4 * np.pi * d_Rx_target ** 2) # (TxN,RxN,chirpN,1,targetN)
Ar = np.sqrt(Pr) # (TxN,RxN,chirpN,1,targetN)

if ComplexSignal:
    frame_data_part = Ar * np.exp(1j * 2 * np.pi * (-fmin * tao + slope / 2 * (2 * tao * t - tao * tao))) # (TxN,RxN,chirpN,rangeN,targetN)
else:
    frame_data_part = Ar * np.cos(1j * 2 * np.pi * (-fmin * tao + slope / 2 * (2 * tao * t - tao * tao)))

frame_data_part = np.sum(frame_data_part, axis=-1) # (TxN,RxN,chirpN,rangeN)
frame_data_part = frame_data_part * np.tile(TxDM_pattern.T,(1,chirpN // len(TxDM_pattern))).reshape(TxN,1,chirpN,1)  # (TxN,RxN,chirpN,rangeN)
frame_data = np.sum(frame_data_part, axis=0)  # (RxN,chirpN,rangeN)
noise_power_dbm = -55
noise = np.random.normal(0, 10 ** (noise_power_dbm / 20), size=(RxN,chirpN,rangeN)) + 1j * np.random.normal(0, 10 ** (noise_power_dbm / 20), size=(RxN,chirpN,rangeN))
frame_data += noise


import matplotlib.pyplot as plt
if TxDM_mode == 'TDM':
    frame_data_ = frame_data.reshape(4,-1,4,256)#.swapaxes(1,2)
    plt.imshow(abs(np.fft.fftshift(np.fft.fftn(frame_data_[0, :, 0, :]),axes=(-2,))))
else:
    frame_data_ = frame_data
    plt.imshow(abs(np.fft.fftshift(np.fft.fftn(frame_data_[0, :, :]), axes=(-2,))))
plt.show()

