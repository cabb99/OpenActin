import numpy as np
from scipy.special import erf

class MDFit:
    def __init__(self, coordinates, experimental_map,n_voxels,voxel_size,padding=3):
        self.coordinates=coordinates
        self.experimental_map=experimental_map
        self.sigma=np.array([1,1,1])
        self.padding=padding
        self.n_voxels=n_voxels
        self.voxel_size=voxel_size
        self.voxel_limits=[np.arange(-padding,n_voxels[0]+1+padding)*voxel_size[0],
                           np.arange(-padding,n_voxels[1]+1+padding)*voxel_size[1],
                           np.arange(-padding,n_voxels[2]+1+padding)*voxel_size[2]]

    def fold_padding(self,volume_map):
        p=self.padding
        vp=volume_map.copy()
        if p>0 and len(volume_map.shape)==3:
            vp[-2*p:-p, :, :] += vp[:p, :, :]
            vp[:, -2*p:-p, :] += vp[:, :p, :]
            vp[:, :, -2*p:-p] += vp[:, :, :p]
            vp[p:2*p, :, :]   += vp[-p:, :, :]
            vp[:, p:2*p, :]   += vp[:, -p:, :]
            vp[:, :, p:2*p]   += vp[:, :, -p:]
            vp=vp[p:-p, p:-p, p:-p]
        elif p>0 and len(volume_map.shape)==4:
            vp[:, -2*p:-p, :, :] += vp[:, :p, :, :]
            vp[:, :, -2*p:-p, :] += vp[:, :, :p, :]
            vp[:, :, :, -2*p:-p] += vp[:, :, :, :p]
            vp[:, p:2*p, :, :]   += vp[:, -p:, :, :]
            vp[:, :, p:2*p, :]   += vp[:, :, -p:, :]
            vp[:, :, :, p:2*p]   += vp[:, :, :, -p:]
            vp=vp[:,p:-p, p:-p, p:-p]
        return vp

    def sim_map(self):
        sigma=self.sigma*np.sqrt(2)
        phix=(1+erf((self.voxel_limits[0]-self.coordinates[:,None,0])/sigma[0]))/2
        phiy=(1+erf((self.voxel_limits[1]-self.coordinates[:,None,1])/sigma[1]))/2
        phiz=(1+erf((self.voxel_limits[2]-self.coordinates[:,None,2])/sigma[2]))/2

        dphix=(phix[:,1:]-phix[:,:-1])
        dphiy=(phiy[:,1:]-phiy[:,:-1])
        dphiz=(phiz[:,1:]-phiz[:,:-1])
        
        smap=(dphix[:,:,None,None]*dphiy[:,None,:,None]*dphiz[:,None,None,:]).sum(axis=0)
        
        return self.fold_padding(smap)

    def corr_coef(self):
        simulation_map=self.sim_map()
        return (simulation_map*self.experimental_map).sum()/np.sqrt((simulation_map**2).sum()*(self.experimental_map**2).sum())

    def dcorr_coef_numerical(self, delta=1e-5):
        num_derivatives = np.zeros(self.coordinates.shape)
        original_corr_coef = self.corr_coef()
    
        for i in range(self.coordinates.shape[0]):
            for j in range(self.coordinates.shape[1]):
                # Perturb coordinates positively
                self.coordinates[i, j] += delta
                positive_corr_coef = self.corr_coef()
                
                # Perturb coordinates negatively
                self.coordinates[i, j] -= 2*delta
                negative_corr_coef = self.corr_coef()
                
                # Compute numerical derivative
                num_derivatives[i, j] = (positive_corr_coef - negative_corr_coef) / (2*delta)
                
                # Reset coordinates to original value
                self.coordinates[i, j] += delta
                
        return num_derivatives

    def fit(self,numerical=False):
        f=1
        for i in range(1000):
            if numerical:
                dx=self.dcorr_coef_numerical()
            else:
                dx=self.dcorr_coef()
            f=.1/np.abs(dx).max()
            self.coordinates=self.coordinates+f*dx
            for j in range(3): 
                self.coordinates[:, j][self.coordinates[:, j] < 0] += self.voxel_size[j] * self.n_voxels[j]
                self.coordinates[:, j][self.coordinates[:, j] >= self.voxel_size[j] * self.n_voxels[j]] -= self.voxel_size[j] * self.n_voxels[j]
            if i%10==0:
                print(i,self.corr_coef())

    def dsim_map_numerical(self, delta=1e-5):
        num_particles = self.coordinates.shape[0]
        sim_map_shape = self.sim_map().shape
        derivatives = {
            'dx': np.zeros((num_particles,) + sim_map_shape),
            'dy': np.zeros((num_particles,) + sim_map_shape),
            'dz': np.zeros((num_particles,) + sim_map_shape)
        }
        
        original_sim_map = self.sim_map()
        
        for i in range(num_particles):
            for j, direction in enumerate(['dx', 'dy', 'dz']):
                original_coordinate = self.coordinates[i, j]
        
                # Perturb coordinate in the positive direction
                self.coordinates[i, j] = original_coordinate + delta
                positive_sim_map = self.sim_map()
        
                # Perturb coordinate in the negative direction
                self.coordinates[i, j] = original_coordinate - delta
                negative_sim_map = self.sim_map()
        
                # Compute numerical derivative for this particle and direction
                derivatives[direction][i] = (positive_sim_map - negative_sim_map) / (2 * delta)
        
                # Reset coordinate to original value
                self.coordinates[i, j] = original_coordinate
        
        return derivatives

    @staticmethod
    def outer_mult(x,y,z):
        return x[:,:,None,None]*y[:,None,:,None]*z[:,None,None,:]
    
    def dsim_map(self):
        sigma=self.sigma*np.sqrt(2)

        x_mu_sigma=(self.voxel_limits[0]-self.coordinates[:,None,0])/sigma[0]
        y_mu_sigma=(self.voxel_limits[1]-self.coordinates[:,None,1])/sigma[1]
        z_mu_sigma=(self.voxel_limits[2]-self.coordinates[:,None,2])/sigma[2]
        
        phix=(1+erf(x_mu_sigma))/2
        phiy=(1+erf(y_mu_sigma))/2
        phiz=(1+erf(z_mu_sigma))/2
        
        dphix_dx= -np.exp(-x_mu_sigma**2) / np.sqrt(np.pi) / sigma[0]
        dphiy_dy= -np.exp(-y_mu_sigma**2) / np.sqrt(np.pi) / sigma[1]
        dphiz_dz= -np.exp(-z_mu_sigma**2) / np.sqrt(np.pi) / sigma[2]
        
        dphix_ds= x_mu_sigma*dphix_dx*np.sqrt(2)
        dphiy_ds= y_mu_sigma*dphiy_dy*np.sqrt(2)
        dphiz_ds= z_mu_sigma*dphiz_dz*np.sqrt(2)

        dphix=(phix[:,1:]-phix[:,:-1])
        dphiy=(phiy[:,1:]-phiy[:,:-1])
        dphiz=(phiz[:,1:]-phiz[:,:-1])
        
        ddphix_dx=dphix_dx[:,1:]-dphix_dx[:,:-1]
        ddphiy_dy=dphiy_dy[:,1:]-dphiy_dy[:,:-1]
        ddphiz_dz=dphiz_dz[:,1:]-dphiz_dz[:,:-1]
        
        ddphix_ds=dphix_ds[:,1:]-dphix_ds[:,:-1]
        ddphiy_ds=dphiy_ds[:,1:]-dphiy_ds[:,:-1]
        ddphiz_ds=dphiz_ds[:,1:]-dphiz_ds[:,:-1]

        dsim={}

        dsim['dx']=self.outer_mult(ddphix_dx,dphiy,dphiz)
        dsim['dy']=self.outer_mult(dphix,ddphiy_dy,dphiz)
        dsim['dz']=self.outer_mult(dphix,dphiy,ddphiz_dz)
        dsim['dsx']=self.outer_mult(ddphix_ds,dphiy,dphiz)
        dsim['dsy']=self.outer_mult(dphix,ddphiy_ds,dphiz)
        dsim['dsz']=self.outer_mult(dphix,dphiy,ddphiz_ds)

        for key in dsim:
            dsim[key]=self.fold_padding(dsim[key])
        return dsim

    def dcorr_coef(self):
        dsim=self.dsim_map()
        dsim=np.array([dsim['dx'],dsim['dy'],dsim['dz'],dsim['dsx'],dsim['dsy'],dsim['dsz']]).transpose(1,0,2,3,4)
        sim=self.sim_map()
        exp=self.experimental_map
        
        #
        num1=np.sum(dsim*exp[None,None,:,:,:],axis=(2,3,4))
        den1=np.sqrt(np.sum(sim**2)) * np.sqrt(np.sum(exp**2))
        
        num2 = np.sum(dsim*sim[None,None,:,:,:],axis=(2,3,4))* np.sum(sim * exp)
        den2 = np.sum(sim**2) * den1
        
        # Final equation
        return ((num1 / den1) - (num2 / den2))[:,:3]

    def test(self):
        assert np.allclose(self.dsim_map()['dx'],self.dsim_map_numerical()['dx'])
        assert np.allclose(self.dsim_map()['dy'],self.dsim_map_numerical()['dy'])
        assert np.allclose(self.dsim_map()['dz'],self.dsim_map_numerical()['dz'])
        assert np.allclose(self.dcorr_coef(),self.dcorr_coef_numerical())

from numba import jit
@jit(nopython=True)
def fold_outer_mult4(x,y,z,padding):
    x_reshaped = x.reshape(x.shape[0], x.shape[1], 1, 1)
    y_reshaped = y.reshape(y.shape[0], 1, y.shape[1], 1)
    z_reshaped = z.reshape(z.shape[0], 1, 1, z.shape[1])
    
    # Perform element-wise multiplication with broadcasting
    vp = x_reshaped * y_reshaped * z_reshaped
    p = padding

    vp[:, -2*p:-p, :, :] += vp[:, :p, :, :]
    vp[:, :, -2*p:-p, :] += vp[:, :, :p, :]
    vp[:, :, :, -2*p:-p] += vp[:, :, :, :p]
    vp[:, p:2*p, :, :]   += vp[:, -p:, :, :]
    vp[:, :, p:2*p, :]   += vp[:, :, -p:, :]
    vp[:, :, :, p:2*p]   += vp[:, :, :, -p:]
    vp = vp[:, p:-p, p:-p, p:-p]
    return vp

@jit(nopython=True)
def numba_erf(x):
    # Preallocate the result array with the same shape as x
    result = np.empty(x.shape, dtype=np.float64)
    # Iterate over all elements in x by their indices
    for index, value in np.ndenumerate(x):
        # Apply math.erf element-wise
        result[index] = math.erf(value)
    return result
    

def dcorr(coordinates,voxel_limits,sigma, experimental_map,padding):
    sigma=self.sigma*np.sqrt(2)

    x_mu_sigma=(voxel_limits[0]-coordinates[:,None,0])/sigma[0]
    y_mu_sigma=(voxel_limits[1]-coordinates[:,None,1])/sigma[1]
    z_mu_sigma=(voxel_limits[2]-coordinates[:,None,2])/sigma[2]
    
    phix=(1+numba_erf(x_mu_sigma))/2
    phiy=(1+numba_erf(y_mu_sigma))/2
    phiz=(1+numba_erf(z_mu_sigma))/2
    
    dphix_dx= -np.exp(-x_mu_sigma**2) / np.sqrt(np.pi) / sigma[0]
    dphiy_dy= -np.exp(-y_mu_sigma**2) / np.sqrt(np.pi) / sigma[1]
    dphiz_dz= -np.exp(-z_mu_sigma**2) / np.sqrt(np.pi) / sigma[2]
    
    dphix_ds= x_mu_sigma*dphix_dx*np.sqrt(2)
    dphiy_ds= y_mu_sigma*dphiy_dy*np.sqrt(2)
    dphiz_ds= z_mu_sigma*dphiz_dz*np.sqrt(2)

    dphix=(phix[:,1:]-phix[:,:-1])
    dphiy=(phiy[:,1:]-phiy[:,:-1])
    dphiz=(phiz[:,1:]-phiz[:,:-1])
    
    ddphix_dx=dphix_dx[:,1:]-dphix_dx[:,:-1]
    ddphiy_dy=dphiy_dy[:,1:]-dphiy_dy[:,:-1]
    ddphiz_dz=dphiz_dz[:,1:]-dphiz_dz[:,:-1]
    
    ddphix_ds=dphix_ds[:,1:]-dphix_ds[:,:-1]
    ddphiy_ds=dphiy_ds[:,1:]-dphiy_ds[:,:-1]
    ddphiz_ds=dphiz_ds[:,1:]-dphiz_ds[:,:-1]

    smap     = fold_outer_mult4(dphix, dphiy, dphiz, padding)
    smap_dx  = fold_outer_mult4(ddphix_dx, dphiy, dphiz, padding)
    smap_dy  = fold_outer_mult4(dphix, ddphiy_dy, dphiz, padding)
    smap_dz  = fold_outer_mult4(dphix, dphiy, ddphiz_dz, padding)
    smap_dsx = fold_outer_mult4(ddphix_ds, dphiy, dphiz, padding)
    smap_dsy = fold_outer_mult4(dphix, ddphiy_ds, dphiz, padding)
    smap_dsz = fold_outer_mult4(dphix, dphiy, ddphiz_ds, padding)

    
    dsim=np.array([smap_dx,smap_dy,smap_dz,smap_dsx,smap_dsy,smap_dsz])
    sim=smap.sum(axis=0)
    exp=experimental_map
    
    num1=np.sum(dsim*exp[None,None,:,:,:],axis=(2,3,4))
    den1=np.sqrt(np.sum(sim**2)) * np.sqrt(np.sum(exp**2))
    
    num2 = np.sum(dsim*sim[None,None,:,:,:],axis=(2,3,4))* np.sum(sim * exp)
    den2 = np.sum(sim**2) * den1
    
    # Final equation
    return ((num1 / den1) - (num2 / den2))[:,:3]

coordinates=np.random.rand(100,3)*(10,20,5)
experimental_map=np.random.rand(10,20,5)


self=MDFit(coordinates,experimental_map,n_voxels=[10,20,5],voxel_size=[1,1,1],padding=4)



