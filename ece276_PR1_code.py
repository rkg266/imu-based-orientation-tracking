import os
import pickle
from load_data import *
import numpy as np
#import jax.numpy as jnp
import autograd.numpy as jnp
from autograd import grad
#from jax import jit, grad
import transforms3d as tf3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

##### USER EDITABLE #####
# CAUTION: 
# 1. train and test data paths must be different. Do not place all files in one directory.
# 2. None of the variables below must be left blank. Especially, provide all the directory paths.
train_imu_path = r'/home/renukrishna/ece276a/Project1/data/imu'
train_cam_path = r'/home/renukrishna/ece276a/Project1/data/cam'
train_vicon_path = r'/home/renukrishna/ece276a/Project1/data/vicon'
test_imu_path = r'/home/renukrishna/ece276a/Project1/testdata/imu'
test_cam_path = r'/home/renukrishna/ece276a/Project1/testdata/cam'

trainpklpath = r'/home/renukrishna/ece276a/Project1/trainpickle' # To store gradient descent parameters for train data
testpklpath = r'/home/renukrishna/ece276a/Project1/testpickle' # To store gradient descent parameters for test data

SavePlotsPath = r'/home/renukrishna/ece276a/Project1/PlotImages'
SavePanoramaPath = r'/home/renukrishna/ece276a/Project1/PanoramaImages'

RUN_TRAINING = 1 # Run gradient descent on train data mentioned in 'TRAIN QUEUE'
RUN_TESTING = 1 # Run gradient descent on test data mentioned in 'TEST QUEUE'
RUN_PLOTTING = 1 # Run plotting for data sets mentioned in 'PLOT_QUEUE'
RUN_PANAROMA = 1 # Run panorama for data sets mentioned in 'PANORAMA_QUEUE'
TRAIN_QUEUE = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TEST_QUEUE = [0, 1] # Should be 0, 1 for the two test datasets - 10, 11
PLOT_QUEUE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 9, 10 are test datasets
PANORAMA_QUEUE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 9, 10 are test datasets
USE_VICON_FOR_PANORAMA = 0 # Whether to use VICON data for panorama 
num_run = 200 # Max iterations for gradient descent
######### USER EDITABLE ENDS ###################

class dataset:
    index=0
    imu=[]
    imu_values=[]
    imu_t=[]
    cam=[]
    cam_t=[]
    vicon=[]
    vicon_t=[]
    vicon_euler=[]  # Angles order z, y, x
    imubias=[]
    qtrn=[]
    qtr2euler=[]
    euler_estmd = []
    AccSensitivity = 300
    GyroSensitivity = 3.33
    Vref = 3300

    def CnvrtIMURaw(self):
        self.imu_values = np.zeros(self.imu.shape)
        s_arr = np.array([0., 0., 0., 0., 0., 0.])
        for i in range(3):
            if i==2:
                s_arr[i] = self.AccSensitivity
            else:
                s_arr[i] = -self.AccSensitivity
        for i in range(3):
            s_arr[i+3] = self.GyroSensitivity * (180/np.pi)
        scale_factor = self.Vref/(1023*s_arr)
        sign_crctn = np.array([-1, -1, 1, 1, 1, 1]) 
        for i in range(self.imu.shape[1]):
            self.imu_values[:, i] = (self.imu[:, i] - self.imubias) * scale_factor
            self.imu_values[:, i] = np.multiply(self.imu_values[:, i], sign_crctn) # Ax, Ay sign flipped
            self.imu_values[2, i] = self.imu_values[2, i] + 1 # Az is 'g' when body is static

    def runMotionModel(self):
        self.qtrn = np.zeros((4,self.imu_t.shape[1]))
        q = [1, 0, 0, 0]
        self.qtrn[:, 0] = q
        for j in range(1, self.imu_t.shape[1]): # Start from second timestamp
            cur_imuvals = self.imu_values[:, j]
            tau = self.imu_t[:, j] - self.imu_t[:, j-1]
            w = np.array([cur_imuvals[4], cur_imuvals[5], cur_imuvals[3]])
            ar = tau*w/2
            ar = np.insert(ar, 0, 0)
            ep = tf3d.quaternions.qexp(ar)
            q = tf3d.quaternions.qmult(q, ep)
            self.qtrn[:, j] = q

    def QuaterToEuler(self):
        self.qtr2euler = np.zeros((3, self.qtrn.shape[1]))
        for i in range(self.qtrn.shape[1]):
            R = tf3d.quaternions.quat2mat(self.qtrn[:, i])
            self.qtr2euler[:, i] = tf3d.euler.mat2euler(R, 'sxyz')

    def gnrttext(self, i, fpath):
        if i==1:
            a = self.imu
            b = self.imu_t
            outstr = ''
            for j in range(b.shape[1]):
                outstr = outstr + str(b[0, j]) + ':\t'
                aa = a[:, j]
                for x in aa:
                    outstr = outstr + str(x) + ', '
                outstr = outstr + '\n'
            fname = 'imuout' + str(self.index) + '.txt'
            tpf = open(os.path.join(fpath, fname), 'w')
            tpf.write(outstr)
            tpf.close

def Qnorm(q):
    nrm = 0
    for i in range(q.shape[0]):
        nrm = nrm + q[i]*q[i]
    return jnp.sqrt(nrm)

def Qexp(q):
    ptrb=1e-6
    q = q + ptrb
    ans = jnp.zeros((4, ))
    qs = q[0]
    qv = q[1:]
    n_qv = jnp.linalg.norm(qv)
    t = jnp.sin(n_qv)/n_qv
    exp_qs = jnp.exp(qs)
    ans = jnp.array([exp_qs*jnp.cos(n_qv), exp_qs*t*qv[0], exp_qs*t*qv[1], exp_qs*t*qv[2]])
    return ans

def Qmult(q, p):
    qs = q[0]
    ps = p[0]
    qv = q[1:]
    pv = p[1:]
    ans = jnp.zeros((4, ))
    tp = qs*pv + ps*qv + jnp.cross(qv, pv)
    ans = jnp.array([qs*ps - jnp.inner(qv, pv), tp[0], tp[1], tp[2]])
    return ans

def Qinv(q):
    ptrb = 1e-6
    q = q + ptrb
    qs = q[0]
    qv = q[1:]
    n_q = jnp.linalg.norm(q)
    qcj = jnp.array([qs, -qv[0], -qv[1], -qv[2]])
    return (qcj/jnp.square(n_q))

def Qlog(q):
    ptrb = 1e-6
    q = q + ptrb
    qs = q[0]
    qv = q[1:]
    n_q = jnp.linalg.norm(q)
    n_qv = jnp.linalg.norm(qv)
    ans = jnp.zeros((4, ))
    tp = jnp.arccos(qs/n_q)/n_qv
    ans = jnp.array([jnp.log(n_q), tp*qv[0], tp*qv[1], tp*qv[2]])
    return ans

#@jit
def MotModel(q, tau, wq): # wq is [0, wx, wy, wz]
    # Jax based motion model
    tp = tau*wq/2
    q_next = Qmult(q, Qexp(tp))
    return q_next 

#@jit
def ObsModel(q):
    # Autograd based observation model
    g = jnp.array([0., 0., 0., 9.8]) # gravity in world frame (z direction should be same as that in intial at)
    tp = Qmult(g, q)
    a = Qmult(Qinv(q), tp)
    return a 

def CostFuncGD(q0, qvect, tauvect, wvect, avect): # All inputs must be arrays of (4, ) quaternions
    motsum = 0
    obssum = 0
    qcur = q0
    g = 9.8
    for i in range(tauvect.shape[0]):
        qnext = qvect[:, i]
        tpmot = Qmult(Qinv(qnext), MotModel(qcur, tauvect[i], wvect[:, i]))
        cur_mot = jnp.square(jnp.linalg.norm(2*Qlog(tpmot)))
        cur_obs = jnp.square(jnp.linalg.norm(avect[:, i+1] - ObsModel(qvect[:, i])/g)) # Check the indexing here again
        motsum = motsum + cur_mot
        obssum = obssum + cur_obs
        qcur = qnext
    cost = 0.5*(motsum + obssum)
    return cost

def sph2cart(r, horiz, vert): #horiz-long, vert-lat
    c = np.pi/2
    x = r*np.sin(c-vert)*np.cos(horiz)
    y = r*np.sin(c-vert)*np.sin(horiz)
    z = r*np.cos(c-vert)
    ans = [x, y, z]
    return ans

def cart2sph(v): 
    r = np.linalg.norm(v)
    x = v[0]
    y = v[1]
    z = v[2]
    r1 = np.sqrt(x*x + y*y)
    if r1 == 0:
        return [r, 0, 0]
    vert = np.arccos(z/r)
    horiz = (y/np.abs(y))*np.arccos(x/r1)
    return [r, horiz, vert]

def cart2sph_vect(P):
    c = np.pi/2
    R = np.linalg.norm(P, axis=0)
    X = P[0, :]
    Y = P[1, :]
    Z = P[2, :]
    R1 = np.sqrt(np.square(R) - np.square(Z))
    sP = np.zeros(P.shape)
    sP[0, :] = R
    sP[1, :] = (Y/np.abs(Y))*np.arccos(X/R1) # horiz
    sP[2, :] = c - np.arccos(Z/R) # vert
    return sP

def ReadData(imu_path, cam_path, vicon_path):
    imu_list = sorted(os.listdir(imu_path))
    cam_list = sorted(os.listdir(cam_path))
    if vicon_path != '':
        vicon_list = sorted(os.listdir(vicon_path))
    else:
        vicon_list = []
    itr = 0
    data_list = []
    for fil in imu_list:
        cur_set = dataset()
        if fil.endswith('.p'):
            t1 = fil.find('imuRaw')
            t2 = fil.find('.p')
            imu_id = int(fil[t1+len('imuRaw'):t2]) - 1
            tp_path = os.path.join(imu_path, fil)
            tp_data = read_data(tp_path)
            cur_set.index = imu_id
            cur_set.imu = tp_data['vals']
            cur_set.imu_t = tp_data['ts']
            data_list.append(cur_set)
            itr = itr + 1
    itr = 0
    for fil in cam_list:
        if fil.endswith('.p'):
            t1 = fil.find('cam')
            t2 = fil.find('.p')
            cam_id = int(fil[t1+len('cam'):t2]) - 1 # All data sets do not have cam data
            # if cam_id > 8:
            #     cam_id = cam_id - 9
            c_id = [x for x in range(len(data_list)) if data_list[x].index == cam_id]
            tp_path = os.path.join(cam_path, fil)
            tp_data = read_data(tp_path)
            data_list[c_id[0]].cam = tp_data['cam']
            data_list[c_id[0]].cam_t = tp_data['ts']
            itr = itr + 1
    itr = 0
    for fil in vicon_list:
        if fil.endswith('.p'):
            t1 = fil.find('viconRot')
            t2 = fil.find('.p')
            vicon_id = int(fil[t1+len('viconRot'):t2]) - 1
            v_id = [x for x in range(len(data_list)) if data_list[x].index == vicon_id]
            tp_path = os.path.join(vicon_path, fil)
            tp_data = read_data(tp_path)
            data_list[v_id[0]].vicon = tp_data['rots']
            data_list[v_id[0]].vicon_t = tp_data['ts']
            itr = itr + 1
    return data_list
###########################################################################################################

# Reading training and testing data
num_train_data = 9
num_test_data = 2
traindata = ReadData(train_imu_path, train_cam_path, train_vicon_path)
testdata = ReadData(test_imu_path, test_cam_path, '')

# IMU bias calculation and values conversion
set1 = traindata[0]
n_idle = 1470 # Start time estimated by visual inspection of vicon plots of 1st training data
meanbias = set1.imu[:, 0:n_idle].mean(1)
for i in range(len(traindata)):  # Assigning the same bias as in 1st set for all the sets
    traindata[i].imubias = meanbias
    traindata[i].CnvrtIMURaw()
for i in range(len(testdata)):
    testdata[i].imubias = meanbias
    testdata[i].CnvrtIMURaw()

# Verifying the data with Qauternion Kinematic Model
for i in range(len(traindata)):
    traindata[i].runMotionModel()
    traindata[i].QuaterToEuler()
for i in range(len(testdata)):
    testdata[i].runMotionModel()
    testdata[i].QuaterToEuler()

if RUN_TRAINING == 1:
    for trid in tqdm(TRAIN_QUEUE):
        qvect = jnp.array(traindata[trid].qtrn)
        qvect = qvect[:, 1:] # remove the first qo=[1, 0, 0, 0]
        #qvect = jnp.zeros((4, traindata[trid].qtrn.shape[1]-1))
        #qvect[0, :] = jnp.ones((1, qvect.shape[1]))
        t1 = traindata[trid].imu_t
        t2 = np.insert(t1, 0, 0)
        t1 = np.append(t1, 0)
        tauvect = t1-t2 # time difference vector
        tauvect = tauvect[1:-1]
        avect0 = traindata[trid].imu_values[0:3, :]
        avect = np.zeros((4, avect0.shape[1]))
        avect[1:, :] = avect0 # linear acceleration vector
        wvect0 = traindata[trid].imu_values[3:, :]
        wvect = np.zeros((4, wvect0.shape[1]))
        wvect[1:, :] = wvect0 # angular velocity vector
        tauvect = jnp.array(tauvect)
        avect = jnp.array(avect)
        wvect = jnp.array(wvect)
        q0 = jnp.array([1., 0., 0., 0.])

        alpha = 1e0
        eps = 1e-2
        count_stop = 0
        cost_store = []
        grad_store = []
        for nr in tqdm(range(num_run)):
            qvect_cur = qvect
            cost1 = CostFuncGD(q0, qvect, tauvect, wvect, avect)
            grad_qvect = grad(CostFuncGD, 1)(q0, qvect, tauvect, wvect, avect)
            qvect = qvect - alpha*grad_qvect
            qvect_norm = jnp.linalg.norm(qvect, axis=0)
            qvect = qvect / qvect_norm
            cost2 = CostFuncGD(q0, qvect, tauvect, wvect, avect)
            grad_nrm = jnp.linalg.norm(grad_qvect)
            if cost2 > cost1:
                alpha = alpha / 10
                qvect = qvect_cur
            else:
                if grad_nrm > 1e-3:
                    alpha = alpha*2
                if cost1 - cost2 < eps:
                    count_stop = count_stop + 1
                else:
                    count_stop = 0
            cost_store.append(cost1)
            grad_store.append(grad_nrm)
            if count_stop == 10:
                break

        cst_name = 'cost'+str(traindata[trid].index)+'.pkl'
        grd_name = 'gradnrm'+str(traindata[trid].index)+'.pkl'
        qvect_name = 'qvect'+str(traindata[trid].index)+'.pkl'
        with open(os.path.join(trainpklpath, cst_name), 'wb') as f:
            pickle.dump(cost_store, f)
        with open(os.path.join(trainpklpath, grd_name), 'wb') as f:
            pickle.dump(grad_store, f)
        with open(os.path.join(trainpklpath, qvect_name), 'wb') as f:
            pickle.dump(qvect, f)

if RUN_TESTING == 1:
    for trid in tqdm(TEST_QUEUE):
        qvect = jnp.array(testdata[trid].qtrn)
        qvect = qvect[:, 1:] # remove the first qo=[1, 0, 0, 0]
        #qvect = jnp.zeros((4, testdata[trid].qtrn.shape[1]-1))
        #qvect[0, :] = jnp.ones((1, qvect.shape[1]))
        t1 = testdata[trid].imu_t
        t2 = np.insert(t1, 0, 0)
        t1 = np.append(t1, 0)
        tauvect = t1-t2 # time difference vector
        tauvect = tauvect[1:-1]
        avect0 = testdata[trid].imu_values[0:3, :]
        avect = np.zeros((4, avect0.shape[1]))
        avect[1:, :] = avect0 # linear acceleration vector
        wvect0 = testdata[trid].imu_values[3:, :]
        wvect = np.zeros((4, wvect0.shape[1]))
        wvect[1:, :] = wvect0 # angular velocity vector
        tauvect = jnp.array(tauvect)
        avect = jnp.array(avect)
        wvect = jnp.array(wvect)
        q0 = jnp.array([1., 0., 0., 0.])

        alpha = 1e0
        eps = 1e-2
        count_stop = 0
        cost_store = []
        grad_store = []
        for nr in tqdm(range(num_run)):
            qvect_cur = qvect
            cost1 = CostFuncGD(q0, qvect, tauvect, wvect, avect)
            grad_qvect = grad(CostFuncGD, 1)(q0, qvect, tauvect, wvect, avect)
            qvect = qvect - alpha*grad_qvect
            qvect_norm = jnp.linalg.norm(qvect, axis=0)
            qvect = qvect / qvect_norm
            cost2 = CostFuncGD(q0, qvect, tauvect, wvect, avect)
            grad_nrm = jnp.linalg.norm(grad_qvect)
            if cost2 > cost1:
                alpha = alpha / 10
                qvect = qvect_cur
            else:
                if grad_nrm > 1e-3:
                    alpha = alpha*2
                if cost1 - cost2 < eps:
                    count_stop = count_stop + 1
                else:
                    count_stop = 0
            cost_store.append(cost1)
            grad_store.append(grad_nrm)
            if count_stop == 10:
                break

        cst_name = 'cost'+str(testdata[trid].index)+'.pkl'
        grd_name = 'gradnrm'+str(testdata[trid].index)+'.pkl'
        qvect_name = 'qvect'+str(testdata[trid].index)+'.pkl'
        with open(os.path.join(testpklpath, cst_name), 'wb') as f:
            pickle.dump(cost_store, f)
        with open(os.path.join(testpklpath, grd_name), 'wb') as f:
            pickle.dump(grad_store, f)
        with open(os.path.join(testpklpath, qvect_name), 'wb') as f:
            pickle.dump(qvect, f)


if RUN_PLOTTING == 1:
    ptrb=1e-6
    for it in PLOT_QUEUE:
        if it < num_train_data:
            cur_set = traindata[it]
            picklePath = trainpklpath
        else:
            cur_set = testdata[it-num_train_data]
            picklePath = testpklpath
        idx = cur_set.index
        cst_name = 'cost'+str(idx)+'.pkl'
        grd_name = 'gradnrm'+str(idx)+'.pkl'
        qvect_name = 'qvect'+str(idx)+'.pkl'
        with open(os.path.join(picklePath, cst_name), 'rb') as f:
            cost_store = pickle.load(f)
        with open(os.path.join(picklePath, grd_name), 'rb') as f:
            grad_store = pickle.load(f)
        with open(os.path.join(picklePath, qvect_name), 'rb') as f:
            qvect_load = pickle.load(f)

        q_est = np.zeros((4, qvect_load.shape[1]+1)) + ptrb
        q_est[0, 0] = 1.
        q_est[:, 1:] = qvect_load
        euler_est = np.zeros((3, q_est.shape[1]))
        for i in range(q_est.shape[1]):
            R_est = tf3d.quaternions.quat2mat(q_est[:, i])
            euler_est[:, i] = tf3d.euler.mat2euler(R_est, 'rzyx')
            #euler_est[:, i] = euler_est[:, i] * np.array([-1, -1, 1])
            euler_est[:, i] = euler_est[:, i] * np.array([1, -1, -1])
        cur_set.euler_estmd = euler_est
        if it < num_train_data:
            cur_set.vicon_euler = np.zeros((3,cur_set.vicon_t.shape[1]))
            for j in range(cur_set.vicon_t.shape[1]):
                rotmatrix = cur_set.vicon[:, :, j]
                cur_set.vicon_euler[:, j] = tf3d.euler.mat2euler(rotmatrix, 'rzyx') # order of angles x, y, z -> roll, pitch, yaw
        # Plot euler anlges
        ptitles = ['Yaw', 'Pitch', 'Roll']
        #ptitles = ['Roll', 'Pitch', 'Yaw']
        fig, ax = plt.subplots(3, 1)
        fig.set_dpi(100)
        fig.set_size_inches(8, 8)
        fig.tight_layout(pad=4.0)
        figtitle = ''
        if it < num_train_data:
            figtitle = figtitle + 'Train_data_' + str(it+1)
        else:
            figtitle = figtitle + 'Test_data_' + str(it+1)
        fig.suptitle(figtitle)
        for p in range(3):
            if it < num_train_data:
                ax[p].plot(list(range(cur_set.vicon_t.shape[1])), list(cur_set.vicon_euler[p, :]), label='Vicon')
            ax[p].plot(list(range(cur_set.imu_t.shape[1])), list(euler_est[p, :]), label='Estimated')
            ax[p].legend()
            ax[p].set_title(ptitles[p])
        #plt.show()
        filname = figtitle + '_plot.jpg' 
        pltpth = os.path.join(SavePlotsPath, filname)
        plt.savefig(pltpth, format='jpg')
        plt.close()

        plt.figure()
        plt.plot(list(range(len(cost_store))), cost_store, label='Cost value')
        plt.title(figtitle+": Cost vs iterations")
        filname = figtitle + '_costplot.jpg' 
        pltpth = os.path.join(SavePlotsPath, filname)
        plt.savefig(pltpth, format='jpg')
        plt.close()
        #plt.show()


# Panaroma latitude-longitude matrix generation
cur_dataset = traindata[0]
camdata = cur_dataset.cam
camtimes = cur_dataset.cam_t
imutimes = cur_dataset.imu_t
M = camdata.shape[0] # rows (vertical down)
N = camdata.shape[1] # columns (horizontal right)
fov_horiz = 60 #degrees - horizontal - longitude
fov_vert = 45 #degrees - vertical - latitude
fov_horiz = (fov_horiz*np.pi)/180 #radians
fov_vert = (fov_vert*np.pi)/180 #radians
f_horiz =  fov_horiz/N
f_vert = fov_vert/M
sph_crd_mat = np.zeros((3, M, N)) # (r, horizAngle, vertAngle)
cart_cam_mat = np.zeros((3, M, N)) # (x-forward, , vertAngle)
#i_down = -int(M/2)
i_down = int(M/2)
for m in range(M): # down
    i_right = int(N/2)
    for n in range(N): # right        
        horiz = f_horiz * i_right
        vert = f_vert * i_down
        sph_crd_mat[0, m, n] = 1
        sph_crd_mat[1, m, n] = horiz
        sph_crd_mat[2, m, n] = vert
        cart_cam_mat[:, m, n] = sph2cart(1, horiz, vert)
        i_right = i_right - 1
        if N%2==0 and i_right==0:
            i_right = i_right - 1
    #i_down = i_down + 1
    i_down = i_down - 1
    if M%2==0 and i_down==0:
        #i_down = i_down + 1
        i_down = i_down - 1

sph = [1, -np.pi/4, np.pi/3]
cart1 = sph2cart(sph[0], sph[1], sph[2])
sph2 = cart2sph(cart1)
cart_cam_mat_reshape = np.reshape(cart_cam_mat, (3, cart_cam_mat.shape[1]*cart_cam_mat.shape[2]))
cart_cam_mat_2sph = cart2sph_vect(cart_cam_mat_reshape)
cart_cam_mat_2sph = np.reshape(cart_cam_mat_2sph, cart_cam_mat.shape)

# Panorama construction
if RUN_PANAROMA == 1:
    ptrb=1e-6
    for it in PANORAMA_QUEUE:
        if it < num_train_data:
            cur_set = traindata[it]
            picklePath = trainpklpath
        else:
            cur_set = testdata[it-num_train_data]
            picklePath = testpklpath
        idx = cur_set.index
        qvect_name = 'qvect'+str(idx)+'.pkl'
        with open(os.path.join(picklePath, qvect_name), 'rb') as f:
            qvect_load = pickle.load(f)
        q_est = np.zeros((4, qvect_load.shape[1]+1)) + ptrb
        q_est[0, 0] = 1.
        q_est[:, 1:] = qvect_load
        rotmat_est = np.zeros((3, 3, q_est.shape[1]))
        for i in range(q_est.shape[1]):
            R_est = tf3d.quaternions.quat2mat(q_est[:, i])
            rotmat_est[:, :, i] = R_est

        if len(cur_set.cam_t) == 0:
            continue
        camdata = cur_set.cam
        camtime = cur_set.cam_t
        camtime = np.reshape(camtime, camtime.shape[1])
        if it < num_train_data:
            vicontime = cur_set.vicon_t
            vicontime = np.reshape(vicontime, vicontime.shape[1])
            vicondata = cur_set.vicon
            cam_nearby_vicon = np.searchsorted(vicontime, camtime)
        imutime = cur_set.imu_t
        imutime = np.reshape(imutime, imutime.shape[1])
        cam_nearby_imu = np.searchsorted(imutime, camtime)

        M0 = 720
        N0 = 1080
        pan_img_mat = np.zeros((M0, N0, 3), dtype='uint8')
        res_horiz_rad = 2*np.pi/N0 # radian per pixel
        #res_vert_rad = 2*np.pi/M0 # radian per pixel
        res_vert_rad = np.pi/M0 # radian per pixel
        pan_img_cntr = np.array([M0/2-1, N0/2-1])
        pan_img_cntr_in_vect = int((M0/2-1)*N0 + (N0/2-1))
        pan_img_vect = np.reshape(pan_img_mat, (M0*N0, 3))

        for cc in tqdm(range(len(camtime))):
            if USE_VICON_FOR_PANORAMA == 1 and it < num_train_data: # only for training data
                if cam_nearby_vicon[cc]-1 < 0:
                    continue
                rotmatrix = vicondata[:, :, cam_nearby_vicon[cc]-1] # Rotation matrix nearest to current cam sample
            else: # Use estimated orientation tracking
                if cam_nearby_imu[cc]-1 < 0:
                    continue
                rotmatrix = rotmat_est[:, :, cam_nearby_imu[cc]-1] # Rotation matrix nearest to current cam sample
            camimg = camdata[:, :, :, cc]
            camimg_vect = np.reshape(camimg, (camimg.shape[0]*camimg.shape[1], 3))
            wrld_cart_cam_mat = np.zeros(cart_cam_mat.shape)
            wrld_sph_cam_mat = np.zeros(cart_cam_mat.shape)   
            
            # Rotation of cam cartesian to world cartesian frame
            cart_cam_mat_reshape = np.reshape(cart_cam_mat, (3, cart_cam_mat.shape[1]*cart_cam_mat.shape[2]))
            wrld_cart_cam_vect = np.matmul(rotmatrix, cart_cam_mat_reshape)
            # Cartesian to spherical
            wrld_sph_cam_vect = np.nan_to_num(cart2sph_vect(wrld_cart_cam_vect)) # nan handling done   
            # Spherical to cylindrical to plane
            horiz_x = wrld_sph_cam_vect[1, :] / res_horiz_rad
            vert_y = wrld_sph_cam_vect[2, :] / res_vert_rad
            pan_x_vect = np.rint(pan_img_cntr[1] - horiz_x).astype(int) # round off to nearest integer
            pan_y_vect = np.rint(pan_img_cntr[0] - vert_y).astype(int)
            pan_vect_ids = np.rint(pan_y_vect*N0 + pan_x_vect).astype(int)
            pan_img_vect[pan_vect_ids, :] = camimg_vect

        pan_img_mat = np.reshape(pan_img_vect, (M0, N0, 3))
        pan_title = ''
        if it < num_train_data:
            pan_title = pan_title + 'Train_data_' + str(it+1)
        else:
            pan_title = pan_title + 'Test_data_' + str(it+1)
        plt.figure()
        plt.axis('off')
        plt.title(pan_title)
        plt.imshow(pan_img_mat)
        #plt.show()
        if USE_VICON_FOR_PANORAMA == 1 and it < num_train_data:
            filname = pan_title + '_vicon_pan.jpg' 
        else:
            filname = pan_title + '_pan.jpg' 
        panpth = os.path.join(SavePanoramaPath, filname)
        plt.savefig(panpth, format='jpg')
bg = 2
