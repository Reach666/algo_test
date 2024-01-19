import numpy as np
# from scipy import ndimage
import imageio # pip install imageio
import matplotlib.pyplot as plt


def pic_compress(k, pic_array):
    u, sigma, vt = np.linalg.svd(pic_array)
    print('pic_array shape:',pic_array.shape)
    print('u shape:', u.shape)
    print('sigma shape:', sigma.shape)
    print('vt shape:', vt.shape)
    plt.figure()
    plt.subplot(121)
    plt.plot(np.log(sigma))
    plt.title('log(Sigma) vs Sigma number')
    plt.subplot(122)
    plt.plot(np.cumsum(sigma))
    plt.ylim(0,1.01*max(np.cumsum(sigma)))
    plt.title('Compression rate vs Sigma number')
    sig = np.eye(k) * sigma[: k]
    new_pic = np.dot(np.dot(u[:, :k], sig), vt[:k, :])  # 还原图像
    # size = u.shape[0] * k + sig.shape[0] * sig.shape[1] + k * vt.shape[1]  # 压缩后大小
    size = u.shape[0] * k + k + k * vt.shape[1]  # 压缩后大小
    return new_pic, size


filename = "./1.jpg"
# ori_img = np.array(imageio.imread(filename, as_gray=True))
ori_img = np.array(imageio.imread(filename, mode='L'))
ori_img[100:105,100:105]=255
ori_img[600:605,700:705]=255
ori_img[1000:1005,100:105]=255
# ori_img_mean = np.mean(ori_img)
# ori_img = ori_img-ori_img_mean
new_img, size = pic_compress(100, ori_img) #30

print("original size:" + str(ori_img.shape[0] * ori_img.shape[1]))
print("compress size:" + str(size))
print("compress rate:",size/(ori_img.shape[0] * ori_img.shape[1]))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(ori_img)
ax[0].set_title("before compress")
ax[1].imshow(new_img)
ax[1].set_title("after compress")
plt.show()
