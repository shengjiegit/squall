#程序用来读取 雷达 二进制 拼图数据
from skimage import morphology,draw,filters,measure,feature
import skimage.filters.rank as sfr
from skimage.morphology import disk
import struct
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.colors as col
import matplotlib.patches as mpathes
import cv2
from scipy import interpolate
from sklearn import metrics
from sklearn.cluster import DBSCAN
def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)
    return contours

def lonlat2ngrid(lon_s,lat_s,lon_n_s,lon_n_e,lat_n_s,lat_n_e,resolution):
    xgrid_s = int((lon_n_s - lon_s) / 0.01)
    xgrid_e = int((lon_n_e - lon_s) / 0.01)
    ygrid_s = int((lat_n_s - lat_s) / 0.01)
    ygrid_e = int((lat_n_e - lat_s) / 0.01)
    nx=xgrid_e - xgrid_s
    ny=ygrid_e - ygrid_s
    return xgrid_s,xgrid_e,ygrid_s,ygrid_e,nx,ny

def grid2lonlat(n,lonlat_start,resolution):
    return lonlat_start + (n-1) * resolution


f = open('O:\squaline\ACHN.CREF000.20160613.184000.matdat', 'rb')
resolution=0.01;nx=6200 ;ny=4200;lon_start=73; lon_end=135; lat_start=12.2;lat_end=54.2 #原始文件建维度
lon_s=115; lon_e=119; lat_s=32;lat_e=40
#裁剪
xgrid_s,xgrid_e,ygrid_s,ygrid_e,NX,NY=lonlat2ngrid(lon_start,lat_start,lon_s,lon_e,lat_s,lat_e,resolution)
bytes=f.read()
dbz=struct.unpack('!26040000B',bytes)
Dbz = np.array(dbz,dtype=float)
Dbz.resize((ny,nx));Dbz=Dbz[::-1,:]

DBZ=Dbz[ygrid_s:ygrid_e,xgrid_s:xgrid_e]
echoo=Dbz[ygrid_s:ygrid_e,xgrid_s:xgrid_e].copy()
#echoo=[ygrid_s:ygrid_e,xgrid_s:xgrid_e])
#Y,X=np.where(DBZ[ygrid_s:ygrid_e,xgrid_s:xgrid_e]>50)
Y,X=np.where(DBZ>45)
vv=list(zip(Y.tolist(),X.tolist()))
db=DBSCAN(eps =20, min_samples =3).fit_predict(vv)
maxdbz=max(set(db.tolist()), key=db.tolist().count)
#dbz=db[db==maxdbz]
Xdbz=X[db==maxdbz]
Ydbz=Y[db==maxdbz]

DBZ[:,:]=0
DBZ[Ydbz,Xdbz]=1
#dn=morphology.binary_dilation(DBZ, selem=morphology.square(60))
#dn=morphology.binary_erosion(dn, selem=morphology.square(5))
#dn=morphology.binary_opening(DBZ, selem=morphology.disk(20))
dn=morphology.binary_closing(DBZ, selem=morphology.disk(20))
#edges = filters.sobel(dn)
#print(edges[np.where(edges>0)])
skeleton =morphology.skeletonize(dn)
#dubianx=measure.subdivide_polygon(dn, degree=2, preserve_ends=False)
#计算中轴和距离变换值
#skel, distance =morphology.medial_axis(dn, return_distance=True)
#中轴上的点到背景像素点的距离
#dist_on_skel = distance * skel
y_center,x_center=np.where(skeleton==True)

x = np.linspace(lon_s, lon_e, NX)
y = np.linspace(lat_s, lat_e, NY)
LON, LAT = np.meshgrid(x, y)
colors_leida = ['#FFFFFF', '#A8DFFD', '#50BEFA', '#01A9F6', '#00EBD9', '#00D500', '#089300', '#FFFF00', '#E9C300' \
    , '#FE9400', '#FF0D00', '#DC0000', '#C40000', '#F400C7', '#9600B4']
cmap_leida = col.ListedColormap(colors_leida)
lat = np.loadtxt('D:/pytho/lat_map.txt')
lon = np.loadtxt('D:/pytho/lon_map.txt')
#dd=plt.pcolor(LON, LAT,echoo)
echo=plt.contourf(LON, LAT, echoo, cmap=cmap_leida)
plt.scatter(grid2lonlat(x_center,lon_s,resolution), grid2lonlat(y_center,lat_s,resolution),marker='o',c="white",edgecolors = 'none')
plt.plot(lon,lat)
plt.colorbar(echo,orientation='vertical')
plt.xlim(lon_s,lon_e)
plt.ylim(lat_s,lat_e)
#plt.imshow(dist_on_skel, interpolation='nearest')
#skimage.measure.find_contours(DBZ, 0.5)
#skel, distance =morphology.medial_axis(DBZ, return_distance=True)
#dist_on_skel = distance * skel
#plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
#plt.imshow(edges, cmap=plt.cm.gray)
plt.show()

'''
DBZ_EAST=DBZ[ygrid_s:ygrid_e,xgrid_s:xgrid_e]

lat = np.loadtxt('D:/pytho/lat_map.txt')
lon = np.loadtxt('D:/pytho/lon_map.txt')
print(DBZ_EAST.shape)
print(Ydbz)
x = np.linspace(lon_e_s, lon_e_e, NX)
y = np.linspace(lat_e_s, lat_e_e, NY)
LON, LAT = np.meshgrid(x, y)
colors_leida = ['#FFFFFF', '#A8DFFD', '#50BEFA', '#01A9F6', '#00EBD9', '#00D500', '#089300', '#FFFF00', '#E9C300' \
    , '#FE9400', '#FF0D00', '#DC0000', '#C40000', '#F400C7', '#9600B4']
cmap_leida = col.ListedColormap(colors_leida)
fig,ax = plt.subplots()
echo=plt.contourf(LON, LAT, DBZ_EAST, cmap=cmap_leida)
plt.plot(lon,lat)
b=get_contour_verts(echo)
#print(lon_s+resolution*(Xdbz))
#plt.scatter(lon_s+resolution*(Xdbz),lat_s+resolution*(Ydbz),c=dbz,marker='.',edgecolors = 'none')
f=list(map(list,zip(Xdbz,Ydbz)))
F = np.array(f,dtype=int)
d=F.reshape(-1,1,2)

#ellipse = cv2.fitEllipse(d)
#el_x=grid2lonlat(ellipse[0][0],lon_s,resolution)
#el_y=grid2lonlat(ellipse[0][1],lat_s,resolution)
#el=mpathes.Ellipse((el_x,el_y),ellipse[1][0]*resolution, ellipse[1][1]*resolution,angle=ellipse[2], linewidth=2, fill=False, zorder=2)
#ax.add_patch(el)
#func = interpolate.interp1d(grid2lonlat(Xdbz,lon_s,resolution), grid2lonlat(Ydbz,lat_s,resolution), kind='cubic')
#xnew=np.arange(min(Xdbz),max(Xdbz),0.01)
#ynew = func(xnew)
#plt.plot(xnew, ynew, 'ro-')
#z1 = np.polyfit(Xdbz, Ydbz,8)
#p1 = np.poly1d(z1)
#plt.scatter(grid2lonlat(Xdbz,lon_s,resolution),grid2lonlat(p1(Xdbz),lat_s,resolution),c="green",edgecolors = 'none')
plt.xlim((110,125))
plt.ylim((30,40))
plt.colorbar(echo,orientation='vertical')
#plt.imshow(DBZ) # 显示图片
plt.show()
#cv2.HoughCircles()
'''