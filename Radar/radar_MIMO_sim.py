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
# frameN = 1

# 天线排布
TxN = 4
RxN = 4
Tx_coord = [[0,0,0],[1*RxN,0,0],[2*RxN,0,0],[3*RxN,0,0]]
Rx_coord = [[0,0,0],[1,0,0],[2,0,0],[3,0,0]]
Rx_d = _lambda / 2
Tx_coord = np.array(Tx_coord)[:TxN] * Rx_d
Rx_coord = np.array(Rx_coord)[:RxN] * Rx_d
# MIMO模式
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
# noise_power = -55 # -50dbm
Pt = 10000

# 目标状态
targetN = 2
target_coord = np.array([[0, 2.5, 0], [0, 2, 0]]) # (targetN,3)
target_speed = np.array([[1, 0, 0], [-1, -1, 0]]) # (targetN,3)
target_rcs = np.array([0.05, 0.05])


### 生成雷达信号 ###
# 计算慢时间坐标
n_chirp = np.arange(chirpN).reshape(chirpN,1,1,1) # (chirpN,1,1,1)
chirp_target_coord = target_coord + target_speed * (0 + Tc * n_chirp) # (chirpN,1,targetN,3)     axis -3 for rangeN
# n_frame = np.arange(frameN).reshape(frameN,1,1,1,1,1,1) # (frameN,1,1,1,1,1,1)
# chirp_target_coord = target_coord + target_speed * (0 + T_frame * n_frame) # (frameN,1,1,chirpN,1,targetN,3)

# 计算相对坐标及飞行时间
Rx_target_coord = chirp_target_coord - Rx_coord.reshape(RxN,1,1,1,3) # (RxN,chirpN,1,targetN,3)
Rx_target_d = np.linalg.norm(Rx_target_coord,axis=-1) # (RxN,chirpN,1,targetN)
Tx_target_coord = chirp_target_coord - Tx_coord.reshape(TxN,1,1,1,1,3) # (TxN,1,chirpN,1,targetN,3)
Tx_target_d = np.linalg.norm(Tx_target_coord,axis=-1) # (TxN,1,chirpN,1,targetN)
tao = (Tx_target_d + Rx_target_d) / c # (TxN,RxN,chirpN,1,targetN)

# # 传统口径天线
# Gr = Gt = 1
# A_Rx = Gr * _lambda ** 2 / (2 * np.pi)
# 矩形贴片天线
antenna_beta = 2 * np.pi / _lambda
antenna_W= _lambda / 4
antenna_L= _lambda / 4
A_Rx = antenna_W * antenna_L
def cart2sph(x,y,z):    # [az,el,r] = cart2sph(x,y,z)
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arcsin(z / r)
    return az,el,r
# Tx天线方向图增益
Tx_target_sph_az, Tx_target_sph_el, Tx_target_sph_r = cart2sph(Tx_target_coord[..., 2], Tx_target_coord[..., 0], Tx_target_coord[..., 1]) # (TxN,1,chirpN,1,targetN)
Tx_target_sph_el = np.pi/2 - Tx_target_sph_el
antenna_f = np.sinc(1/np.pi * antenna_beta * antenna_W/2 * np.sin(Tx_target_sph_el) * np.sin(Tx_target_sph_az)) \
            * np.cos(antenna_beta * antenna_L/2 * np.sin(Tx_target_sph_el) * np.cos(Tx_target_sph_az))
Gt = np.sqrt((np.cos(Tx_target_sph_az) * antenna_f) ** 2 + (np.cos(Tx_target_sph_el) * np.sin(Tx_target_sph_az) * antenna_f) ** 2) # (TxN,1,chirpN,1,targetN)
# Rx天线方向图增益
Rx_target_sph_az, Rx_target_sph_el, Rx_target_sph_r = cart2sph(Rx_target_coord[..., 2], Rx_target_coord[..., 0], Rx_target_coord[..., 1]) # (RxN,chirpN,1,targetN)
Rx_target_sph_el = np.pi/2 - Rx_target_sph_el
antenna_f = np.sinc(1/np.pi * antenna_beta * antenna_W/2 * np.sin(Rx_target_sph_el) * np.sin(Rx_target_sph_az)) \
            * np.cos(antenna_beta * antenna_L/2 * np.sin(Rx_target_sph_el) * np.cos(Rx_target_sph_az))
Gr = np.sqrt((np.cos(Rx_target_sph_az) * antenna_f) ** 2 + (np.cos(Rx_target_sph_el) * np.sin(Rx_target_sph_az) * antenna_f) ** 2) # (RxN,chirpN,1,targetN)
# 计算信号幅度
RCS = target_rcs # (targetN)
Ptar = Pt * Gt * RCS / (4 * np.pi * Tx_target_d ** 2) # (TxN,1,chirpN,1,targetN)
Pr = Ptar * Gr * A_Rx / (4 * np.pi * Rx_target_d ** 2) # (TxN,RxN,chirpN,1,targetN)
Ar = np.sqrt(Pr) # (TxN,RxN,chirpN,1,targetN)

# 合成雷达信号
t = np.arange(0, Ts * rangeN, Ts).reshape(rangeN,1) # (rangeN,1)
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

