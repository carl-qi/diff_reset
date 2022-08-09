import matplotlib.pyplot as plt
import pickle 
x = pickle.load(open('remote_traj.pkl', 'rb'), encoding='latin1')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(140, -90)
ax.set_xlim3d(0.2, 0.8)
ax.set_ylim3d(0, 0.5)

ax.set_zlim3d(0.2, 0.8)

pcl1 = x['obs'][0, :1000, :]


ax.scatter(pcl1[:,0], pcl1[:,1], pcl1[:,2], marker='o', alpha=0.5, s=2)
pcl2 = x['obs'][0, 1000:1100, :]
ax.scatter(pcl2[:,0], pcl2[:,1], pcl2[:,2], marker='o', alpha=0.5, s=2)
pcl = x['obs'][0, 1100:, :]
ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], marker='x', alpha=0.5, s=2)
plt.show()
