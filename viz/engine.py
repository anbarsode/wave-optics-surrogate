"""Calculations for the local Gravitational Lensing visualizer.

Coordinates use the point-mass Einstein angle from the redshifted mass and
Planck18 distances.  GLoW supplies both I(t) and its regularised FFT: no
time-domain curve in this module is made by manually Fourier transforming F(w).
"""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from astropy.constants import G, M_sun, c
from astropy.cosmology import Planck18
from contourpy import contour_generator
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bgo.expansion import amplification as bgo_amplification

MODELS = {
    "PM": "Point mass", "SIS": "Singular isothermal sphere", "SIE": "Singular isothermal ellipsoid",
    "CIS": "Cored isothermal sphere", "NFW": "Navarro–Frenk–White", "eNFW": "Elliptical NFW",
    "EPL": "Elliptical power law", "Hernquist": "Hernquist",
}
GLOW_MODELS = {"PM", "SIS", "SIE", "CIS", "NFW"}
BGO_MODELS = {"PM", "SIS"}


@dataclass(frozen=True)
class Settings:
    model: str
    source: np.ndarray
    use_external: bool
    convergence: float
    shear: float
    lens_redshift: float
    source_redshift: float
    redshifted_mass: float
    axis_ratio: float
    core_radius: float
    scale_radius: float
    concentration: float
    slope: float

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> "Settings":
        value = request.get("parameters", {})
        model = request.get("model", "PM")
        lens_redshift = float(value.get("z_lens", .5))
        source_redshift = max(float(value.get("z_source", 1.5)), lens_redshift + .01)
        use_external = bool(value.get("use_external", False))
        convergence = max(0., float(value.get("convergence", 0.))) if use_external else 0.
        # Positive projected mass gives kappa >= 0.  gamma < 1-kappa keeps the
        # far-field mapping non-critical instead of extending multiple images to infinity.
        shear = min(max(0., float(value.get("shear", 0.))), max(0., .98-convergence)) if use_external else 0.
        return cls(model if model in MODELS else "PM",
                   np.array([float(value.get("impact_x", .5)), float(value.get("impact_y", 0.))]),
                   use_external, convergence, shear, lens_redshift, source_redshift,
                   max(float(value.get("mass", 100.)), 1e-3),
                   np.clip(float(value.get("axis_ratio", .8)), .2, 1.),
                   np.clip(float(value.get("core_radius", .08)), .005, 1.),
                   np.clip(float(value.get("scale_radius", .3)), .02, 4.),
                   np.clip(float(value.get("concentration", 8.)), 2., 30.),
                   np.clip(float(value.get("slope", 2.)), 1.2, 2.8))

    @property
    def mass_seconds(self) -> float:
        """4 G M_Lz / c^3, so dimensionless frequency is w=2 pi f t_M."""
        return float(4*G.value*self.redshifted_mass*M_sun.value/c.value**3)


def _json(values: np.ndarray) -> list[float]:
    return np.nan_to_num(np.asarray(values), nan=0., posinf=1e12, neginf=-1e12).astype(float).ravel().tolist()


def _curve(values: np.ndarray, label: str, time: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    values = np.asarray(values, complex)
    output: dict[str, Any] = {"label": label, "real": _json(values.real), "imag": _json(values.imag),
                              "amplitude": _json(abs(values)), "phase": _json(np.unwrap(np.angle(values)))}
    if time is not None:
        output["time"], output["time_response"] = _json(time[0]), _json(time[1])
    return output


class Profile:
    """Dimensionless, axis-aligned profile potential with optional external field."""

    def __init__(self, settings: Settings, maximum_radius: float):
        self.s, self.maximum_radius = settings, maximum_radius
        self.h_radius = self.h_potential = None
        if settings.model == "Hernquist":
            radius = np.geomspace(1e-7, maximum_radius*2+5, 8000)
            self.h_radius = radius
            self.h_potential = cumulative_trapezoid(self._hernquist_alpha(radius), radius, initial=0.)

    def _nfw(self, radius: np.ndarray) -> np.ndarray:
        from glow import lenses
        lens = lenses.Psi_NFW({"psi0": self.s.concentration/8., "xs": self.s.scale_radius})
        return lens.psi_x(np.maximum(radius, 1e-9))

    def _hernquist_alpha(self, radius: np.ndarray) -> np.ndarray:
        """Projected Hernquist deflection alpha=(1-F)/x, with x=r/a."""
        x = np.maximum(np.asarray(radius)/self.s.scale_radius, 1e-8)
        lower, upper = x < 1-1e-7, x > 1+1e-7
        function = np.full_like(x, 1/3)
        function[lower] = np.arccosh(1/x[lower])/np.sqrt(1-x[lower]**2)
        function[upper] = np.arccos(1/x[upper])/np.sqrt(x[upper]**2-1)
        return 1.5*(1-function)/x

    def psi(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        r = np.hypot(x1, x2)
        if self.s.model == "PM": value = np.log(np.maximum(r, 1e-8))
        elif self.s.model == "SIS": value = r
        elif self.s.model == "SIE": value = np.sqrt(self.s.axis_ratio*x1**2+x2**2/self.s.axis_ratio)
        elif self.s.model == "CIS": value = np.sqrt(r**2+self.s.core_radius**2)
        elif self.s.model == "NFW": value = self._nfw(r)
        elif self.s.model == "eNFW": value = self._nfw(np.sqrt(self.s.axis_ratio*x1**2+x2**2/self.s.axis_ratio))
        elif self.s.model == "EPL":
            radius = np.sqrt(self.s.axis_ratio*x1**2+x2**2/self.s.axis_ratio)
            exponent = 3-self.s.slope
            value = radius**exponent/exponent
        else:
            value = np.interp(r, self.h_radius, self.h_potential)
        if self.s.use_external:
            value = value + .5*self.s.convergence*(x1**2+x2**2) + .5*self.s.shear*(x1**2-x2**2)
        return value

    def gradient(self, x: np.ndarray) -> np.ndarray:
        step = 2e-5*max(1., np.linalg.norm(x)); x1, x2 = x
        return np.array([(self.psi(x1+step, x2)-self.psi(x1-step, x2))/(2*step),
                         (self.psi(x1, x2+step)-self.psi(x1, x2-step))/(2*step)], float)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        step = 4e-4*max(1., np.linalg.norm(x)); x1, x2 = x; f = self.psi
        h11 = (f(x1+step,x2)-2*f(x1,x2)+f(x1-step,x2))/step**2
        h22 = (f(x1,x2+step)-2*f(x1,x2)+f(x1,x2-step))/step**2
        h12 = (f(x1+step,x2+step)-f(x1+step,x2-step)-f(x1-step,x2+step)+f(x1-step,x2-step))/(4*step**2)
        return np.array([[h11,h12],[h12,h22]], float)


def _solve_images(profile: Profile, settings: Settings, span: float) -> list[dict[str, Any]]:
    """Stationary points from a seed grid, classified by the Fermat Hessian."""
    images: list[dict[str, Any]] = []
    for x1 in np.linspace(-span, span, 13):
        for x2 in np.linspace(-span, span, 13):
            solution = root(lambda x: x-profile.gradient(x)-settings.source, [x1, x2])
            if not solution.success or np.linalg.norm(solution.fun) > 2e-6 or np.linalg.norm(solution.x) > 3*span:
                continue
            # The logarithmic point-mass potential is singular at the origin.
            # Finite differencing creates a nonphysical stationary point there.
            if settings.model == "PM" and np.linalg.norm(solution.x) < 1e-4:
                continue
            if any(np.linalg.norm(solution.x-item["x"]) < 2e-4 for item in images): continue
            matrix = np.eye(2)-profile.hessian(solution.x); determinant = float(np.linalg.det(matrix))
            eigenvalue, eigenvector = np.linalg.eigh(matrix); morse = int(np.count_nonzero(eigenvalue < 0))
            stretch = 1/np.maximum(abs(eigenvalue), 1e-5); major = int(np.argmax(stretch))
            images.append({"x": solution.x, "tau": float(.5*np.sum((solution.x-settings.source)**2)-profile.psi(*solution.x)),
                           "determinant": determinant, "morse": morse,
                           "shape": {"angle": float(np.arctan2(eigenvector[1,major], eigenvector[0,major])),
                                     "major": float(np.clip(stretch[major],.35,8)), "minor": float(np.clip(stretch[1-major],.35,8))}})
    return sorted(images, key=lambda item: item["tau"])


def _segments(axis: np.ndarray, values: np.ndarray, levels: list[float]) -> list[dict[str, Any]]:
    maker, answer = contour_generator(x=axis, y=axis, z=values), []
    for level in levels:
        for line in maker.lines(level):
            if len(line) > 1: answer.append({"level": float(level), "points": np.asarray(line)[::2].tolist()})
    return answer


def _field(profile: Profile, settings: Settings, images: list[dict[str, Any]], resolution: int) -> dict[str, Any]:
    span = profile.maximum_radius; axis = np.linspace(-span, span, resolution); x1, x2 = np.meshgrid(axis, axis); spacing = axis[1]-axis[0]
    psi = profile.psi(x1,x2); gy,gx = np.gradient(psi, spacing, spacing, edge_order=2)
    h11 = np.gradient(gx,spacing,axis=1,edge_order=2); h22 = np.gradient(gy,spacing,axis=0,edge_order=2)
    h12 = .5*(np.gradient(gx,spacing,axis=0,edge_order=2)+np.gradient(gy,spacing,axis=1,edge_order=2))
    sy,sx = x2-gy, x1-gx; determinant = (1-h11)*(1-h22)-h12**2
    magnification = np.divide(1.,determinant,out=np.zeros_like(determinant),where=abs(determinant)>1e-10)
    delay = .5*((x1-settings.source[0])**2+(x2-settings.source[1])**2)-psi; reference = images[0]["tau"]
    residual = np.hypot(settings.source[0]-sx, settings.source[1]-sy); critical = _segments(axis, determinant, [0.]); caustic=[]
    for segment in critical:
        points=np.asarray(segment["points"]); ix=np.clip(np.searchsorted(axis,points[:,0]),0,resolution-1); iy=np.clip(np.searchsorted(axis,points[:,1]),0,resolution-1)
        caustic.append({"level":0.,"points":np.column_stack((points[:,0]-gx[iy,ix],points[:,1]-gy[iy,ix])).tolist()})
    negative=determinant<0; stride=max(3,resolution//22); cells=[]
    for row in range(0,resolution-stride,stride):
        for column in range(0,resolution-stride,stride):
            corners=((row,column),(row,column+stride),(row+stride,column+stride),(row+stride,column))
            if any(negative[i,j] for i,j in corners): cells.append([[float(sx[i,j]),float(sy[i,j])] for i,j in corners])
    # The multiple-image region has to be measured in the source plane, not
    # by simply adding its lens-plane pixels.  Map every negative-parity cell
    # to its four source-plane corners and add its (absolute) polygon area.
    # This is the finite-cell version of dA_source = |det A| dA_lens.
    source_area = 0.
    for row in range(resolution-1):
        for column in range(resolution-1):
            if np.mean(negative[row:row+2, column:column+2]) < .5:
                continue
            polygon = np.array([[sx[row,column], sy[row,column]], [sx[row,column+1], sy[row,column+1]],
                                [sx[row+1,column+1], sy[row+1,column+1]], [sx[row+1,column], sy[row+1,column]]])
            source_area += .5*abs(np.dot(polygon[:,0], np.roll(polygon[:,1], -1)) - np.dot(polygon[:,1], np.roll(polygon[:,0], -1)))
    maximum=max((image["tau"]-reference for image in images),default=.25); edge=bool(np.any(negative[[0,-1],:]) or np.any(negative[:,[0,-1]]))
    return {"extent":[-span,span,-span,span],"resolution":resolution,"delay":_json(delay-reference),
            "log_magnification":_json(np.clip(np.log10(abs(magnification)+1e-10),-1,3)),"negative":negative.astype(np.uint8).ravel().tolist(),
            "residual":_json(np.clip(np.log10(residual+1e-6),-6,1)),"critical":critical,"caustic":caustic,"mapped_cells":cells,
            "delay_contours":_segments(axis,delay-reference,np.linspace(0,max(.25,2*maximum),7).tolist()),
            "multiple_image_radius":np.inf if edge else float(np.sqrt(source_area/np.pi)),"multiple_image_infinite":edge}


def _geometric(w: np.ndarray, images: list[dict[str, Any]]) -> np.ndarray:
    answer=np.zeros_like(w,complex); reference=images[0]["tau"]
    for item in images: answer += np.exp(1j*w*(item["tau"]-reference)-.5j*np.pi*item["morse"])/np.sqrt(abs(item["determinant"]))
    return answer


def _glow_lens(settings: Settings):
    """Rotate the whole axis-aligned system into GLoW's source-axis frame."""
    if settings.model not in GLOW_MODELS: raise ValueError(f"GLoW has no {MODELS[settings.model]} lens class")
    from glow import lenses
    phi=np.arctan2(settings.source[1],settings.source[0]); y=float(np.hypot(*settings.source))
    if settings.model=="PM": base=lenses.Psi_PointLens({"psi0":1.})
    elif settings.model=="SIS": base=lenses.Psi_SIS({"psi0":1.})
    elif settings.model=="SIE": base=lenses.Psi_eSIS({"psi0":1.,"q":settings.axis_ratio,"alpha":-phi})
    elif settings.model=="CIS": base=lenses.Psi_CIS({"psi0":1.,"rc":settings.core_radius})
    else: base=lenses.Psi_NFW({"psi0":settings.concentration/8.,"xs":settings.scale_radius})
    if not settings.use_external: return base,y
    ext=lenses.Psi_Ext({"kappa":settings.convergence,"gamma1":settings.shear*np.cos(2*phi),"gamma2":-settings.shear*np.sin(2*phi)})
    return lenses.CombinedLens({"lenses":[base,ext]}),y


def _glow(settings: Settings,w: np.ndarray,display_points: int,accuracy: float,images: list[dict[str,Any]]) -> tuple[np.ndarray,tuple[np.ndarray,np.ndarray],str]:
    """Reference GLoW response plus regularised FFT; paired resolutions check convergence."""
    from glow import freq_domain_c,time_domain_c
    lens,y=_glow_lens(settings); tolerance=10**accuracy
    levels=(16000,20000) if tolerance<=.008 else ((8000,12000) if tolerance<=.03 else (3000,4500)); values=[]; responses=[]
    four_pm=settings.model=="PM" and settings.use_external and len(images)>=4
    for nt in levels:
        if four_pm:
            response=time_domain_c.It_AreaIntegral_C(lens,y,{"n_rho":40000 if nt==levels[0] else 80000,"n_theta":4000,"Nt":20000,"tmax":1000,"parallel":False})
        else:
            integral=time_domain_c.It_SingleIntegral_C if lens.isAxisym else time_domain_c.It_MultiContour_C
            response=integral(lens,y,{"Nt":nt,"tmax":1e4,"parallel":False})
        transform=freq_domain_c.Fw_FFT_C(response,{"wmin":float(min(w)),"wmax":float(1.2*max(w)),"eval_mode":"exact","parallel":False,"N_above_discard":7})
        values.append(np.asarray(transform(w),complex)); responses.append(response)
    delta=np.linalg.norm(values[-1]-values[-2])/max(np.linalg.norm(values[-1]),1e-20); time,response=responses[-1].t_grid,responses[-1].It_grid
    take=np.unique(np.linspace(0,len(time)-1,display_points).astype(int)); return values[-1],(time[take],response[take]),f"GLoW FFT, Nt={levels[-1]} (Δ={delta:.2g})"


def _bgo(settings: Settings,w: np.ndarray,order: int,images: list[dict[str,Any]]) -> tuple[np.ndarray,str]:
    if settings.model not in BGO_MODELS: return _geometric(w,images),"Geometric series, K=0"
    return bgo_amplification(w,settings.model,settings.source,strength=1.,shear=(settings.shear,0.),convergence=settings.convergence,order=order),f"BGO series, K={order}"


def _analytic(settings: Settings,w: np.ndarray) -> np.ndarray:
    from glow import freq_domain_c
    y=float(np.hypot(*settings.source))
    if settings.model=="PM" and not settings.use_external: return np.asarray(freq_domain_c.Fw_AnalyticPointLens_C(y,{"parallel":False})(w),complex)
    if settings.model=="SIS" and not settings.use_external: return np.asarray(freq_domain_c.Fw_SemiAnalyticSIS_C(y,{"parallel":False})(w),complex)
    raise ValueError("analytic solver is available for isolated point-mass and SIS lenses")


def _cosmology(settings: Settings,field: dict[str,Any]) -> dict[str,Any]:
    dl=Planck18.angular_diameter_distance(settings.lens_redshift).value; ds=Planck18.angular_diameter_distance(settings.source_redshift).value; dls=Planck18.angular_diameter_distance_z1z2(settings.lens_redshift,settings.source_redshift).value
    mass=settings.redshifted_mass/(1+settings.lens_redshift)*M_sun.value; mpc=3.085677581491367e22; scale=np.sqrt(4*G.value*mass/c.value**2*dls*mpc/(dl*mpc*ds*mpc)); radius=field["multiple_image_radius"]
    return {"theta_mi_mas":float(np.degrees(scale*radius)*3.6e6) if np.isfinite(radius) else np.inf,"theta_mi_infinite":bool(field["multiple_image_infinite"]),"lens_time_seconds":settings.mass_seconds}


def calculate(request: dict[str,Any]) -> dict[str,Any]:
    settings=Settings.from_request(request); option=request.get("solver_settings",{}); points=int(np.clip(option.get("frequency_samples",128),32,1024)); grid=int(np.clip(option.get("sky_resolution",128),32,1024)); times=int(np.clip(option.get("time_samples",128),32,1024)); order=int(np.clip(option.get("bgo_order",0),0,10)); accuracy=float(np.clip(option.get("glow_accuracy",-2.),-3.,-1.)); p=request.get("parameters",{})
    fmin=max(float(p.get("f_min",10.)),1e-3); fmax=max(float(p.get("f_max",1000.)),fmin*1.01); frequency=np.geomspace(fmin,fmax,points); w=np.maximum(2*np.pi*frequency*settings.mass_seconds,1e-5)
    span=max(3.,1.8+1.4*np.linalg.norm(settings.source)+2*settings.scale_radius); profile=Profile(settings,span); images=_solve_images(profile,settings,span)
    if not images: raise ValueError("No stationary images were found; try a weaker external field")
    field=_field(profile,settings,images,grid); curves={}; messages=[]
    for name in request.get("solvers") or ["glow","bgo"]:
        try:
            if name=="glow": value,time,label=_glow(settings,w,times,accuracy,images); curves[name]=_curve(value,label,time)
            elif name=="bgo": value,label=_bgo(settings,w,order,images); curves[name]=_curve(value,label)
            elif name=="analytic": curves[name]=_curve(_analytic(settings,w),"Analytic radial solution")
        except Exception as error: messages.append(f"{name}: {error}")
    reference=images[0]["tau"]; names=("minimum","saddle","maximum"); rows=[]
    for index,item in enumerate(images,1):
        magnification=1/item["determinant"]; rows.append({"order":index,"x":item["x"].tolist(),"magnification":float(magnification),"log_magnification":float(np.clip(np.log10(abs(magnification)),-1,3)),"type":names[min(item["morse"],2)],"delay_dimensionless":float(item["tau"]-reference),"delay_seconds":float((item["tau"]-reference)*settings.mass_seconds),"shape":item["shape"]})
    return {"model":settings.model,"model_name":MODELS[settings.model],"frequency_hz":_json(frequency),"w":_json(w),"curves":curves,"images":rows,"source":settings.source.tolist(),"field":field,"cosmology":_cosmology(settings,field),"messages":messages,"glow_accuracy":{"target_relative_l2":10**accuracy,"nominal_seconds":float(np.clip(.25/10**accuracy,.5,8.))}}
