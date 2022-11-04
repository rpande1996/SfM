import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

input_data = r"../input/"

data = loadmat(input_data + 'tracks.mat')
nan_x = data['track_x']
nan_y = data['track_y']


def replaceNaN(X):
    new_x = X.copy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i, j]):
                new_array = X.copy()
                new_array1 = new_array[i]
                lst = [x for x in new_array1 if not np.isnan(x)]
                max_val = lst[-1]
                new_array1[np.isnan(new_array1)] = max_val
                new_x[i] = new_array1
    return new_x


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix / norm
    return matrix

def genD(x, y):
    D = []
    for i in range(x.shape[0]):
        temp1 = x[i].tolist()
        temp2 = y[i].tolist()
        temp = np.vstack((temp1, temp2))
        if len(D) != 0:
            D = np.vstack((D, temp))
        else:
            D = temp

    D = np.asarray(D)
    return D

def factorizeD(D):
    U, S1, Vt = np.linalg.svd(D)

    W = np.zeros((U.shape[1], Vt.shape[0]))
    W[:S1.size, :S1.size] = np.diag(S1)

    u3 = U[:, 0:3]
    v3 = Vt.T[:, 0:3]
    w3 = W[0:3, 0:3]
    return u3, v3, w3

def EliminateAffine(A):
    Q = []
    R = []

    for i in range(int(A.shape[0] / 2)):
        A_i = A[i:i + 2, :]

        a = A_i[0][0]
        b = A_i[0][1]
        c = A_i[0][2]

        d = A_i[1][0]
        e = A_i[1][1]
        f = A_i[1][2]

        t1 = [a * a, a * b + b * a, a * c + c * a, b * b, c * b + b * c, c * c]
        t2 = [d * d, d * e + e * d, d * f + f * d, e * e, f * e + e * f, f * f]
        t3 = [a * d, a * e + b * d, a * f + c * d, b * e, b * f + e * c, c * f]
        q_t = np.vstack((t1, np.vstack((t2, t3))))

        r_t = np.vstack((1, np.vstack((1, 0))))

        if len(Q) != 0 and len(R) != 0:
            Q = np.vstack((Q, q_t))
            R = np.vstack((R, r_t))
        else:
            Q = q_t
            R = r_t

    l = np.linalg.pinv(Q) @ R
    l = (l.T).tolist()
    l = l[0]
    L = np.array([[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]])
    C = np.linalg.cholesky(L)

    return C


def affineSFM(nan_x, nan_y):
    track_x = replaceNaN(nan_x)
    track_y = replaceNaN(nan_y)
    centered_x = (track_x - track_x.mean(axis=0, keepdims=True)).T
    centered_y = (track_y - track_y.mean(axis=0, keepdims=True)).T
    D = genD(centered_x, centered_y)

    u, v, w = factorizeD(D)

    A_hat = u @ (w ** 0.5)
    X_hat = (w ** 0.5) @ (v.T)

    C = EliminateAffine(A_hat)

    M = A_hat @ C
    S = np.linalg.inv(C) @ X_hat

    X1 = S[0, :]
    Y1 = S[1, :]
    Z1 = S[2, :]

    cam_pos = np.zeros((int(M.shape[0] / 2), 3))
    for i in range(int(M.shape[0] / 2)):
        c = M[2 * i:2 * (i + 1), :]
        c1 = c[0]
        c2 = c[1]
        kf = np.cross(c1, c2)
        cam_pos[i] = kf / np.linalg.norm(kf)
    cam_pos = normalize_2d(cam_pos)
    cX = cam_pos[:, 0]
    cY = cam_pos[:, 1]
    cZ = cam_pos[:, 2]

    return X1, Y1, Z1, cX, cY, cZ

out = "../output/"
x, y, z, cx, cy, cz = affineSFM(nan_x, nan_y)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, color='red', s=3)
ax.view_init(-67,44)
ax.set_title("Structure from Motion")
plt.savefig(out + "SfM.jpg")
plt.show()

plt.plot(cx, cy)
plt.title("XY Plane")
plt.savefig(out + "XY_CameraPath.jpg")
plt.show()

plt.plot(cy, cz)
plt.title("YZ Plane")
plt.savefig(out + "YZ_CameraPath.jpg")
plt.show()

plt.plot(cx, cz)
plt.title("XZ Plane")
plt.savefig(out + "XZ_CameraPath.jpg")
plt.show()