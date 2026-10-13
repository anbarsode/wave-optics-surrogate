import numpy as np
from itertools import product
from contourpy import contour_generator
from scipy.interpolate import interp1d, CubicSpline

class SLML:
    """
    Class defining a point lens in an external potential at the origin.
    
    Parameters
    ----------
    y1 : float
        Source-plane coordinate in the x1 direction.
    y2 : float, optional
        Source-plane coordinate in the x2 direction (default: 0.).
    k : float
        Convergence value due to external potential.
    g1 : float
        Shear due to external potential along the x1 direction.
    g2 : float, optional
        Shear due to external potential along the x2 direction (defalut: 0.).
    mratio : float, optional
        Ratio of the microlens mass to the effective lensing mass scale M0
        (default: 1.).
    
    Attributes
    ----------
    img : ndarray, shape (N, 2)
        Cartesian coordinates of the images found by ``image_locations``.
        Found and stored during initialization.
    img_td_abs : ndarray, shape (N,)
        Absolute time delays at the image locations.
    img_td : ndarray, shape (N,)
        Time delays at the image locations (relative to the minimum if
        multiple images are present).
    img_mag : ndarray, shape (N,)
        Magnification at each image location.
    img_midx : ndarray or float
        Morse index (image classification) for each image.
    img_macro : ndarray, shape (N, 2)
        Cartesian coordinates of the macro image location.
    img_macro_td : ndarray, shape (N,)
        Absolute time delay of the macro image.
    img_macro_mag : ndarray, shape (N,)
        Magnification of the macro image.
    """

    def __init__(self, k, g1, mratio, y1, y2=0, g2=0.):
        self.y1 = y1
        self.y2 = y2
        self.kappa = k
        self.gamma1 = g1 
        self.gamma2 = g2
        self.mass_ratio = mratio
        self.img_macro, self.img_macro_td, self.img_macro_mag = self.macro_image()
        self.img = self.image_locations()
        self.img_td_abs = self.time_delay(*self.img.T)
        idxs = np.argsort(self.img_td_abs)
        self.img = self.img[idxs]
        self.img_td_abs = self.img_td_abs[idxs]
        self.img_td = self.img_td_abs - np.min(self.img_td_abs)
        self.img_mag = self.magnification(*self.img.T)
        self.img_midx = self.morse_index()
    
    def potential_macrolens(self, x1, x2):
        """
        Returns the lensing potential for the constant kappa-gamma
        model without any micro-lenses.
        """
        xsq = x1**2 + x2**2
        xsq_diff = x1**2 - x2**2
        return 0.5 * (self.kappa * xsq + self.gamma1 * xsq_diff + 2. * self.gamma2 * x1 * x2)
    
    def potential_microlens(self, x1, x2):
        """
        Returns the lensing potential due to the micro-lens.
        """
        xsq = x1**2 + x2**2
        return self.mass_ratio * 0.5 * np.log(xsq)
    
    def potential(self, x1, x2):
        return self.potential_macrolens(x1, x2) + self.potential_microlens(x1, x2)
    
    def time_delay(self, x1, x2, minimize=False):
        """
        Returns the absolute time delay at lens plane coordinate (x1, x2)
        """
        td_ = 0.5 * ((x2 - self.y2) ** 2 + (x1 - self.y1) ** 2) - self.potential(x1, x2)
        if minimize:
            td_ -= td_.min()
        return td_
    
    def alpha(self, x1, x2):
        """Deflection angle as a function of the coordinates (x1, x2)"""
        xsq = x1**2 + x2**2
        a1 = x1 * (self.kappa + self.gamma1 + self.mass_ratio/xsq) + self.gamma2*x2
        a2 = x2 * (self.kappa - self.gamma1 + self.mass_ratio/xsq) + self.gamma2*x1
        return a1, a2
    
    def grad_time_delay(self, x1, x2):
        """Returns gradient of time delay as function of coordinates (x1, x2)"""
        xsq = x1**2 + x2**2
        grad = np.empty(np.shape(x1) + (2,))
        grad[..., 0] = (
            -x1 * (self.kappa + self.gamma1 + self.mass_ratio/xsq - 1)
            - x2*self.gamma2 - self.y1
        )
        grad[..., 1] = (
            -x2 * (self.kappa - self.gamma1 + self.mass_ratio/xsq - 1)
            - x1*self.gamma2 - self.y2
        )
        return grad

    def gamma_effective(self, x1, x2):
        """Returns the effective shear"""
        xsq = x1**2 + x2**2
        xsq_diff = x1**2 - x2**2
        g1 = self.gamma1 - (xsq_diff * self.mass_ratio / xsq**2)
        g2 = self.gamma2 - (2 * self.mass_ratio * x1 * x2 / xsq**2)
        return g1, g2

    def hessian(self, x1, x2):
        """Returns the Hessian matrix associated with the time delay map at (x1, x2)."""
        g1, g2 = self.gamma_effective(x1, x2)
        A11 = 1 - self.kappa - g1
        A22 = 1 - self.kappa + g1
        A12 = -g2
        H = np.empty(np.shape(A11) + (2,2))
        H[...,0,0] = A11
        H[...,0,1] = A12
        H[...,1,0] = A12
        H[...,1,1] = A22
        return H
    
    def morse_index(self):
        """
        Returns the Morse Index depending on the type
        of image, given the image location x1im, x2im.
        """
        tr = 2*(1-self.kappa)
        det = 1/self.img_mag
        midx = np.full(len(det), -1.)
        midx[det < 0] = 0.5
        midx[(det > 0) & (tr > 0)] = 0
        midx[(det > 0) & (tr < 0)] = 1
        return midx
        
    def magnification(self, x1, x2):
        """Returns the magnification at (x1, x2)"""
        H = self.hessian(x1, x2)
        det = (
            H[...,0,0]*H[...,1,1]
            - H[...,0,1]*H[...,1,0]
        )
        return 1.0/det
        
    def lens_equation(self, vars):
        x1, x2 = vars
        eq1 = -self.y1 + x1 * (1 - self.kappa - self.gamma1) - self.gamma2 * x2 - (self.mass_ratio * x1 / (x1**2 + x2**2))
        eq2 = -self.y2 + x2 * (1 - self.kappa + self.gamma1) - self.gamma2 * x1 - (self.mass_ratio * x2 / (x1**2 + x2**2))
        return [eq1, eq2]
            
    def macro_image(self):
        """ Returns the image location of the macro image """
        theta = 0.5 * np.arctan2(self.gamma2,self.gamma1)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        gamma = np.sqrt(self.gamma1**2 + self.gamma2**2)
        D = - self.y1 * cos_theta - self.y2 * sin_theta
        E = self.y1 * sin_theta - self.y2 * cos_theta
        A = (1 - self.kappa - gamma) / 2
        C = (1 - self.kappa + gamma) / 2
        x1_image = (- D / 2 / A) * cos_theta - (- E / 2 / C) * sin_theta
        x2_image = (- D / 2 / A) * sin_theta + (- E / 2 / C) * cos_theta
        macro_td = 0.5 * ((x2_image - self.y2) ** 2 + 
                          (x1_image - self.y1) ** 2) - self.potential(x1_image, x2_image)
        macro_mag = 1/((1 - self.kappa)**2 - gamma**2)
        return [x1_image, x2_image], macro_td, macro_mag

    @staticmethod
    def is_duplicate(point, points_list, tol=1e-3):
        """ Check if point is within tol of any point in points_list. """
        for p in points_list:
            if np.linalg.norm(point - p) < tol:
                return True
        return False

    def image_locations(self, search_window=4, num_init=10, 
                        tol=1e-6, max_iter=100):
        x_range = np.linspace(-search_window/2, search_window/2, num_init)
        X = np.array(list(product(x_range, x_range)), dtype=float)
        for _ in range(max_iter):
            grad = self.grad_time_delay(X[:, 0], X[:, 1])
            grad_norm = np.linalg.norm(grad, axis=1)
            converged = grad_norm < tol
            if np.all(converged):
                break
            H = self.hessian(X[:, 0],X[:, 1])
            active = ~converged
            delta = np.linalg.solve(H[active],grad[active])
            X[active] -= delta
        roots = X[converged]
        found_points = []
        for sp in roots:
            if not self.is_duplicate(sp, found_points, tol=1e-2):
                found_points.append(sp)
        return np.asarray(found_points)


def Ftilde_contour(lens, t_max, dt, dx_fac=1, verbose=False, dx1=None, dx2=None, t_min=0):
    """ 
    Computing the raw F(t) 
    for SLML lens using contour summation method 
    Params:
    ---------------
    lens: object of a SLML lens class
    t_max: maximum time delay to be resolved (relative to global minima)
    dt: bin width of histogram
    """
    assert len(lens.img)!=0
    minima_td = min(lens.img_td_abs)
    t_max += minima_td
    bounds_x1, bounds_x2 = _compute_bounds(t_max, lens, eps=0.1)
    dtdx1, dtdx2 = lens.grad_time_delay(bounds_x1[0], bounds_x2[0]).T
    if not dx1:
        dx1 = dt / np.abs(dtdx1) * dx_fac
    if not dx2:
        dx2 = dt / np.abs(dtdx2) * dx_fac
    x1_ = np.arange(bounds_x1[0], bounds_x1[1], dx1)
    x2_ = np.arange(bounds_x2[0], bounds_x2[1], dx2)
    X1, X2 = np.meshgrid(x1_, x2_)
    n = int((t_max-minima_td) / dt)
    Tarray = np.logspace(0, np.log10(t_max+1), n)
    Tarray -=1
    Tarray = Tarray + t_min + minima_td
    td_map = lens.time_delay(X1, X2)
    cg = contour_generator(x=x1_,y=x2_,z=td_map)
    level_coords = [cg.lines(level) for level in Tarray]
    Ft = []
    for level in level_coords:
        res_sum = 0
        for seg in level:
            if seg.shape[0] < 2:
                continue
            dx = np.diff(seg[:,0])
            dy = np.diff(seg[:,1])
            ds = np.hypot(dx, dy)
            grad = lens.grad_time_delay(seg[:,0], seg[:,1])
            gradnorm = np.hypot(grad[:,0], grad[:,1])[:-1]
            res_sum += np.sum(ds / gradnorm) /2. / np.pi
        Ft.append(res_sum)
    
    Tarray-=minima_td
    Ft[0] = np.sqrt(lens.img_mag[np.argmin(lens.img_td)])
    if verbose:
        print(f"x1 bounds = {bounds_x1} | x2 bounds = {bounds_x2} | t_max = {t_max} | dx1 = {dx1}, dx2 = {dx2}")
        print(f"Number of td samples = {len(Tarray)}")
    return Tarray, np.array(Ft)


def _compute_bounds(T, lens, eps=1e-1):
    """
    Returns the bounds of the computational domain for F(t)
    params:
    -------------------------
    T: absolute time to be resolved
    lens: object of class SLML
    """
    
    gamma = np.sqrt(lens.gamma1**2 + lens.gamma2**2)
    y = np.sqrt(lens.y1**2 + lens.y2**2)
    
    A = (1 - lens.kappa - gamma) / 2
    C = (1 - lens.kappa + gamma) / 2
    discriminant = A * C
    
    if abs(discriminant) > 1e-12: # condition for ellipse

        # Rotation matrix components
        theta = 0.5 * np.arctan2(lens.gamma2,lens.gamma1)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Compute the constants for the bounds
        D = - lens.y1 * cos_theta - lens.y2 * sin_theta
        E = lens.y1 * sin_theta - lens.y2 * cos_theta
        F = (y**2)/2 - T
        
        # Calculate the semi-major and semi-minor axes
        common_term = (D**2/4/A) + (E**2/4/C) - F
        a = np.sqrt(common_term / A) * (1 + eps)
        b = np.sqrt(common_term / C) * (1 + eps)

        # Center of ellipse
        x1c = (- D / 2 / A)
        x2c = (- E / 2 / C)
        x1_center = x1c * cos_theta - x2c * sin_theta
        x2_center = x1c * sin_theta + x2c * cos_theta

        t = np.linspace(0, 2*np.pi, 1000)  # parameter t for ellipse points
        
        # Parametric equations for rotated ellipse
        x_1 = x1_center + a * cos_theta * np.cos(t) - b * sin_theta * np.sin(t)
        x_2 = x2_center + a * sin_theta * np.cos(t) + b * cos_theta * np.sin(t)
        
        # Calculate the bounds of the rectangle
        x1_min, x1_max = np.min(x_1), np.max(x_1)
        x2_min, x2_max = np.min(x_2), np.max(x_2)
                
        return (x1_min, x1_max), (x2_min, x2_max)
    else:
        raise ValueError("The provided values do not form an ellipse.")

def get_training_data(times, y1, y2, k, phi_g, mratio=1, w_max=30,
                      t_max=100, dt_fac=2.5, dx=0.05):
    """ Returns the transformed Ft on which
    we train our surrogate model """
    g = k
    g1 = -g * np.cos(2*phi_g)
    g2 = -g * np.sin(2*phi_g)
    # create the lens instance
    lens = SLML(y1=y1, y2=y2, k=k, mratio=mratio, g1=g1, g2=g2)
    dt = np.pi / w_max / dt_fac
    # computing F(t)
    t_vals, Ft_vals = Ftilde_contour(lens, t_max, dt, 
                                     dx1=dx, dx2=dx, verbose=False)
    Ft_interped = interp1d(t_vals, Ft_vals, fill_value='extrapolate')(times)
    # get the transformed F(t)
    Ft_transformed = _get_transformed_Ft(lens, Ft_interped, times)
    # Find window to remove the discontinuity in the transformed F(t)
    # then we replace the data within the window by interpolated data
    window_list = _find_peak_window(lens, times, Ft_interped,
                                   peak_window_frac=0.05)
    mask_window = np.zeros_like(times, dtype=bool)
    for window in window_list:
        left_idx, right_idx, estimated_peak = window
        tmin = times[left_idx]
        tmax = times[right_idx]
        mask_window |= (times >= tmin) & (times <= tmax)
    # Points outside the window
    t_outside = times[~mask_window]
    Ft_outside = Ft_transformed[~mask_window]
    # interpolating within the window
    cs = CubicSpline(t_outside, Ft_outside)
    Ft_transformed[mask_window] = cs(times[mask_window])
    return lens, Ft_interped, Ft_transformed

def _step_function(x):
    """
    Returns the standard Heaviside step function:
    f(x) = 0 for x < 0
    f(x) = 1 for x >= 0
    """
    return np.where(x < 0, 0.0, 1.0)

def _get_F_singular(lens, t):
    """
    This returns the contribution of the singular parts to F(t).
    We use this function to construct the transformed Ft.
    """
    eps = 1e-200 # use this to avoid blow up of the log function below
    min_idx = np.where(np.isclose(lens.img_midx, 0))[0] # get minima indices
    sad_idx = np.where(np.isclose(lens.img_midx, 0.5))[0] # get saddle indices
    max_idx = np.where(np.isclose(lens.img_midx, 1))[0] # get maxima indices
    F_reduced = np.zeros_like(t)
    CM, Cm = 0., 0.
    # the singular contributions below are computed from 
    # Ulmer & Goodman 1995
    if len(min_idx) > 0: # check if there is atleast one minima
        for idx in min_idx:
            # contribution from each minima to the singular Ft
            F_reduced += np.sqrt(lens.img_mag[idx]) * _step_function(t-lens.img_td[idx])
            Cm += np.sqrt(lens.img_mag[idx])
    if len(sad_idx) > 0: # check if there is atleast one saddle
        for idx in sad_idx:
            # contribution from each saddle to the singular Ft
            F_reduced -= np.sqrt(-lens.img_mag[idx])\
                * np.log(np.abs(t-lens.img_td[idx]) + eps) / np.pi
    if len(max_idx) > 0: # check if there is atleast one maxima
        for idx in max_idx:
            # contribution from each maxima to the singular Ft
            F_reduced += np.sqrt(lens.img_mag[idx]) * _step_function(lens.img_td[idx]-t)
            CM += np.sqrt(lens.img_mag[idx])
    # contribution to the global DC Ft by the external potential
    macro_contribution = np.sqrt(lens.img_macro_mag)
    F_reduced += 1 - Cm # this needs modification in the presence of a maxima image.
    return F_reduced

def _get_transformed_Ft(lens, Ft, t):
    """
    Contribution from the regular part of F(t).
    """
    # singular contribution
    F_reduced = _get_F_singular(lens, t)
    # removing the singular contribution
    return Ft - F_reduced

def _find_peak_window(lens, t, Ft, peak_window_frac=0.05):
    # getting all image time delays
    # mask = (lens.img_mag < 0) & (np.abs(lens.img_mag) < 50)
    mask = (lens.img_td >= 1e-6) & (np.abs(lens.img_mag) < 50)
    all_peaks = lens.img_td[mask]
    win_list = []
    for t_peak in all_peaks:
        if t_peak > 1e-10:
            t_start = (1- peak_window_frac) * t_peak
            t_end = (1 + peak_window_frac) * t_peak
            start_of_window = np.where(t >= t_start)[0][0]
            end_of_window = np.where(t >= t_end)[0][0]
            real_peak_loc = np.where(t >= t_peak)[0][0]
            win_list.append((start_of_window, end_of_window, real_peak_loc))
    return win_list

def get_backtransformed_Ft(lens, Ftt, t):
    """
    Combining contribution from singular and nonsingular F(t).
    """
    # singular contribution
    F_reduced = _get_F_singular(lens, t)
    # adding to the non-singular part
    return Ftt + F_reduced


class ReconstructPeak:
    """
    Class for reconstructing peak of F(t).
    """
    def __init__(self, lens, t, Ft, peak_window_frac=0.05):
        self.lens = lens
        self.t = t
        self.Ft = Ft
        self.peak_window_frac = peak_window_frac

    def get_approx_Ft(self):
        assert len(self.lens.img)!=0
        image_timedelays = self.lens.img_td
        Ft = np.zeros_like(self.t)
        for ii in range(len(self.lens.img_td)):
            img_morse_idx = self.lens.img_midx[ii]
            if img_morse_idx == 0.5:
                Ft -= (np.sqrt(np.abs(self.lens.img_mag[ii])) * 
                       np.log(np.abs(image_timedelays[ii]-self.t)) / np.pi)
        return np.asarray(Ft)

    def find_peak_window(self):
        # getting saddle image Tj
        mask = (self.lens.img_mag < 0) & (np.abs(self.lens.img_mag) < 50)
        all_peaks = self.lens.img_td[mask]
        win_list = []
        for t_peak in all_peaks:
            if t_peak > 1e-10:
                t_start = (1- self.peak_window_frac) * t_peak
                t_end = (1 + self.peak_window_frac) * t_peak
                start_of_window = np.where(self.t >= t_start)[0][0]
                end_of_window = np.where(self.t >= t_end)[0][0]
                real_peak_loc = np.where(self.t >= t_peak)[0][0]
                win_list.append((start_of_window, end_of_window, real_peak_loc))
        return win_list

    def get_peak_reconstructed_Ft(self):
        Ft_approx = self.get_approx_Ft()
        Ft_corrected = np.copy(self.Ft)
        peak_window_finder = self.find_peak_window()
        for ii in range(len(peak_window_finder)):
            left_idx, right_idx, estimated_peak = peak_window_finder[ii]
            # Get the difference between the Ft and Ft_approx on the left and right
            diff_left = self.Ft[left_idx] - Ft_approx[left_idx]
            diff_right = self.Ft[right_idx] - Ft_approx[right_idx]
            # Rescale the left and right part
            Ft_approx_rescaled_left = Ft_approx[:estimated_peak] + diff_left
            Ft_approx_rescaled_right = Ft_approx[estimated_peak:] + diff_right
            # combine to get peak corrected Ft
            Ft_corrected[left_idx : estimated_peak] = Ft_approx_rescaled_left[left_idx : estimated_peak]
            Ft_corrected[estimated_peak : right_idx+1] = Ft_approx_rescaled_right[0 : right_idx + 1 - estimated_peak]
            Ft_corrected[estimated_peak] = 0.5 * (Ft_corrected[estimated_peak - 1] + Ft_corrected[estimated_peak + 1])
        return Ft_corrected

def ampFac(t, Ft):
    """
    Returns the angular frequency (dimensionless)
    and amplification factor
    Input params:
    ---------------
    t: time array
    Ft: time series data
    """
    fft_ = np.fft.rfft(Ft) * (t[1] - t[0])
    freq_ = np.fft.rfftfreq(len(t), t[1] - t[0]) * 2.0 * np.pi
    ampfac = fft_ * freq_ * 1j 
    ampfac = np.conj(ampfac) * np.exp(1j * freq_ * t[0])
    ampfac += Ft[-1]
    return freq_, ampfac

'''
W_MAX = 30
w_vals, Fw_vals = ampFac(t_dense, Ft_recons)
idxs = (w_vals > 0) & (w_vals <= W_MAX)
w_vals = w_vals[idxs]
Fw_vals = Fw_vals[idxs]

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(w_vals, np.abs(Fw_vals))
plt.xscale('log')
plt.subplot(122)
plt.plot(w_vals, np.angle(Fw_vals))
plt.xscale('log')
'''