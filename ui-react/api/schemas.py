"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class GeoParams(BaseModel):
    Ks: float = 1.5e-5
    theta_s: float = 0.45
    theta_r: float = 0.05
    alpha: float = 0.8
    n: float = 1.4
    l: float = 0.5
    c_prime: float = 5.0
    phi_prime: float = 28.0
    gamma: float = 20.0
    beta: float = 42.0
    gamma_w: float = 9.81


class NormParams(BaseModel):
    t_max: float = 123.0
    z_max: float = 55.0
    psi_min: float = -500.0
    psi_max: float = -46.907


class PredictRequest(BaseModel):
    z: list[float]
    t: list[float]
    norm: Optional[NormParams] = None


class GridRequest(BaseModel):
    z_min: float = 0.5
    z_max: float = 40.0
    z_res: int = 50
    t_min: float = 0.0
    t_max: float = 123.0
    t_res: int = 50
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None


class FSRequest(BaseModel):
    z: list[float]
    t: list[float]
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None


class VGRequest(BaseModel):
    psi_min: float = -500.0
    psi_max: float = -0.01
    n_points: int = 300
    alpha: float = 0.8
    n: float = 1.4
    theta_s: float = 0.45
    theta_r: float = 0.05
    Ks: float = 1.5e-5
    l: float = 0.5


class ComparisonRequest(BaseModel):
    times: list[float]
    norm: Optional[NormParams] = None


class PDEResidualRequest(BaseModel):
    z_min: float = 0.5
    z_max: float = 40.0
    z_res: int = 30
    t_min: float = 0.0
    t_max: float = 123.0
    t_res: int = 30
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None


class MonteCarloRequest(BaseModel):
    z_min: float = 0.5
    z_max: float = 40.0
    z_res: int = 80
    time: float = 96.0
    n_samples: int = 200
    cov_frac: float = 0.10
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None


class CriticalSlipRequest(BaseModel):
    times: list[float]
    z_res: int = 200
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None


class SensitivityRequest(BaseModel):
    z: float = 10.0
    t: float = 96.0
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None


class ExportRequest(BaseModel):
    z_min: float = 0.5
    z_max: float = 40.0
    z_res: int = 50
    t_min: float = 0.0
    t_max: float = 123.0
    t_res: int = 50
    geo: Optional[GeoParams] = None
    norm: Optional[NormParams] = None
