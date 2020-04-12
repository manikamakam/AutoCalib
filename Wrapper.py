import numpy as np
import glob
import cv2
import scipy.linalg as la
import math
from scipy.optimize import least_squares


def get_corners(images, world_corners):
	corner_pts =[]
	H = np.zeros([3,3])
	for img in images:
		# img = cv2.imread(img_path)

		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


		ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
		
		corner_pts.append(corners)

		# cv2.circle(resized,(corners[0][0][0],corners[0][0][1]),20,[0,0,255])              # world point [21.5,21.5]
		# cv2.circle(resized,(corners[8][0][0],corners[8][0][1]),20,[0,0,255])              # world point [193.5,21.5]
		# cv2.circle(resized,(corners[53][0][0],corners[53][0][1]),20,[0,0,255])            # world point [193.5,129]
		# cv2.circle(resized,(corners[45][0][0],corners[45][0][1]),20,[0,0,255])            # world point [21.5,129]

		dst= np.array([corners[0][0], corners[8][0], corners[53][0], corners[45][0]],dtype='float32')

		hom, status = cv2.findHomography(world_corners,dst)
		# print(hom.dtype)
		H  = np.dstack([H,hom])

	# print(H.shape)
	H = H[:,:,1:]
	H = np.array(H,dtype='float32')
	# print(H.shape)
	# print(H.dtype)
	# print(len(corner_pts[0]))
	corner_pts = np.array(corner_pts,dtype='float32')
	# print(corner_pts.shape)
	return H, corner_pts

def v(H,i,j):
	v = [H[0][i]*H[0][j], H[0][i]*H[1][j] + H[0][j]*H[1][i], H[1][i]*H[1][j], H[2][i]*H[0][j] + H[0][i]*H[2][j], 
		H[2][i]*H[1][j] + H[1][i]*H[2][j], H[2][i]*H[2][j]]
	v = np.array(v,dtype='float32')

	return v


def get_K(H):
	V = []
	for i in range(H.shape[2]):

		temp = v(H[:,:,i],0,1)
		temp1 = v(H[:,:,i],0,0)-v(H[:,:,i],1,1)
		V.append(temp)
		V.append(temp1)

        # print(V_n.shape)
	V = np.array(V,dtype='float32')
	# print(V.shape)

	U,S,Vh = np.linalg.svd(V)

	# print(U.shape)
	# print(S)
	# print(V.shape)

	b = Vh[-1,:]
	# print(b)

	B = np.zeros([3,3])
	B[0][0]=b[0]
	B[0][1]=b[1]
	B[1][0]=b[1]
	B[0][2]=b[3]
	B[2][0]=b[3]
	B[1][1]=b[2]
	B[1][2]=b[4]
	B[2][1]=b[4]
	B[2][2]=b[5]

	# print(B.dtype)

	v0 = (B[0][1]*B[0][2] - B[0][0]*B[1][2])/(B[0][0]*B[1][1]-B[0][1]**2)

	lam = (B[2][2]-((B[0][2]**2)+v0*(B[0][1]*B[0][2] - B[0][0]*B[1][2]))/B[0][0])

	alpha = math.sqrt(lam/B[0][0])
	beta = math.sqrt((lam*B[0][0])/(B[0][0]*B[1][1] - B[0][1]**2))
    # print(beta)
	gamma = -(B[0][1])*(alpha**2)*beta/lam
	u0 = (gamma*v0/beta) - B[0][2]*(alpha**2)/lam

	K = np.array([[alpha, gamma, u0],
                  [0,   beta,  v0],
                  [0,   0,      1]],dtype='float32')
	# print(K)
	# print(K.dtype)
	return K

def get_extrinsic_params(K,H):
    Kinv = np.linalg.inv(K)
  
    lam_inv = (np.linalg.norm(np.matmul(Kinv,H[:,0])) +np.linalg.norm(np.matmul(Kinv,H[:,1])))/2
    # print("lam", 1/lam)

    sgn = np.linalg.det(np.matmul(Kinv,H))
    if sgn<0:
        s = np.matmul(Kinv,H)*-1/lam_inv
    elif sgn>=0:
        s = np.matmul(Kinv,H)/lam_inv

    # s = np.matmul(Kinv,H)/lam_inv

    r1 = s[:,0]
    r2 = s[:,1]
    # print(r1.shape)
    # print(r2.shape)
    r3 = np.cross(r1,r2)
    t = s[:,2]
    t = np.reshape(t,(3,1))
#     t = t/t[2]
    # print(r1)
    Q = np.array([r1,r2,r3],dtype='float32').T

    u,s,v =np.linalg.svd(Q)
    R = np.matmul(u,v)
    # print(R.shape)
    # print(t.shape)
    # print(np.hstack([R,t]))
    Rt = np.hstack([R,t])
    Rt =np.array(Rt,dtype='float32')
    return Rt

def reproj_error(init, corner_pts, H):
	K = np.zeros((3,3))
	dist = np.zeros((2,1))

	K[0][0] = init[0]
	K[1][1] = init[1]
	K[0][1] = init[2]
	K[0][2] = init[3]
	K[1][2] = init[4]
	K[2][2] = 1

	dist[0] = init[5]
	dist[1] = init[6]

	world_pts = []
	for i in range(6):
		for j in range(9):
			world_pts.append([21.5*(j+1),21.5*(i+1),0,1])

	world_pts = np.array(world_pts,dtype='float32')
	world_pts = world_pts.T
	error =[]
	for i in range(H.shape[2]):
		# print("iiiiii",i)

		Rt = get_extrinsic_params(K,H[:,:,i])
		# print(Rt.shape)

		P = np.matmul(K, Rt)
		norm_pts = np.matmul(Rt, world_pts)
		norm_pts = norm_pts/norm_pts[2]
		img_pts = np.matmul(P, world_pts)
		
		img_pts = img_pts/img_pts[2]
		
		# print(img_pts[0].shape)
		u_hat = img_pts[0] + (img_pts[0] - K[0][2])*(dist[0]*(norm_pts[0]**2 + norm_pts[1]**2) + dist[1]*(norm_pts[0]**2 + norm_pts[1]**2)**2)
		v_hat = img_pts[1] + (img_pts[1] - K[1][2])*(dist[0]*(norm_pts[0]**2 + norm_pts[1]**2) + dist[1]*(norm_pts[0]**2 + norm_pts[1]**2)**2)

		reproj = np.vstack([u_hat, v_hat])
		# print(reproj.shape)
		reproj = reproj.T

		pts = corner_pts[i, :, 0, :]
		# print(pts.shape)
		# diff = np.subtract(reproj, corner_pts[i])
		diff = np.linalg.norm(np.subtract(pts,reproj),axis=1)**2

		error.append(diff)

	error = np.array(error,dtype='float32') 
	error = error.reshape(702)

	# print(error.shape)	

	return error

def rms_error(K,D,H,corner_pts):

	world_pts = []
	for i in range(6):
		for j in range(9):
			world_pts.append([21.5*(j+1),21.5*(i+1),0,1])

	world_pts = np.array(world_pts,dtype='float32')

	mean = 0
	error = np.zeros([2,1])
	for i in range(H.shape[2]):
		Rt = get_extrinsic_params(K,H[:,:,i])
		print(world_pts.dtype)
		print(Rt.dtype)
		print(K.dtype)
		print(D.dtype)

		img_pts, _ = cv2.projectPoints(world_pts, Rt[:,0:3], Rt[:,3], K, D)

		img_pts = np.array(img_pts,dtype='float32')
		print(img_pts.shape)
		errors = np.linalg.norm(corner_pts[i,:,0,:]-img_pts[:,0,:],axis=1)
		print(errors.shape)
		error = np.concatenate([error,np.reshape(errors,(errors.shape[0],1))])
		# error = np.mean(errors)
		# mean = error+mean
	mean_error = np.mean(error)
	return mean_error




paths = glob.glob("Calibration_Imgs/*")
images= []
for path in paths:
	img = cv2.imread(path)
	images.append(img)

clone_images=[]
for img in images:
    clone_images.append(img.copy())

world_corners = np.array([[21.5, 21.5], [21.5*9,21.5], [21.5*9, 21.5*6], [21.5,21.5*6]], dtype='float32')
H, corner_pts = get_corners(images, world_corners)

c=0
for img in images:
	for i in range(corner_pts.shape[1]):
		cv2.circle(img,(int(corner_pts[c][i][0][0]),int(corner_pts[c][i][0][1])),5, [0,0,255],5) 
	h, w, _= img.shape
	dim = (int(0.4*w), int(0.4*h))
	cv2.imwrite("outputs/original_images/"+ str(c)+ ".png",img)
	# cv2.imshow("original image", cv2.resize(img, dim))

	# if cv2.waitKey(00)==ord('q'):
	# 	cv2.destroyAllWindows()
	# 	break
	c=c+1


np.set_printoptions(formatter={'float_kind':'{:f}'.format})
K = get_K(H)
print("Initial Estimate of Intrinsic Camera Matrix:")
print(K)
# get_extrinsic_params(K,H[:][:][0])

init = [K[0][0],K[1][1],K[0][1],K[0][2],K[1][2],0,0]	

# print(init)
# reproj_error(init, corner_pts, H)
res = least_squares(reproj_error,x0=np.squeeze(init),method='lm',args=(corner_pts,H))
# print(res.x)

K = np.zeros((3,3))
D = np.zeros((2,1))

K[0][0] = res.x[0]
K[1][1] = res.x[1]
K[0][1] = res.x[2]
K[0][2] = res.x[3]
K[1][2] = res.x[4]
K[2][2] = 1

D[0] = res.x[5]
D[1] = res.x[6]

print("\n Final Intrinsic Camera Matrix:")
print(K)
print("\n Distortion parameters:")
print(D)

# print(K.dtype)
K=np.array(K,dtype='float32')
D = np.array([D[0],D[1],0,0,0],dtype='float32')
# print(D.dtype)

undist_images = []
for img in clone_images:
    undist = cv2.undistort(img,K,D)
    undist_images.append(undist)

_,undist_corners = get_corners(undist_images,world_corners)

c=0
for img in undist_images:
	for i in range(undist_corners.shape[1]):
		cv2.circle(img,(int(undist_corners[c][i][0][0]),int(undist_corners[c][i][0][1])),5, [255,0,0],5) 
	h, w, _= img.shape
	dim = (int(0.4*w), int(0.4*h))
	cv2.imwrite("outputs/undistorted_images/"+ str(c)+ ".png",img)
	c=c+1

error = rms_error(K,D,H,corner_pts)










# corner_pts.extend(corners)



