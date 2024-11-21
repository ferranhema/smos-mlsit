#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
from scipy.optimize import root
from permittivity_models import epsilon_Vant_ice, epsilon_snow, epsilon_water
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function to compute the brightness temperature using the radiative transfer model of N layers from Burke et al 1979. The function receives the incidence angle, 
the ice thickness, the ice temperature, the snow thickness, the snow presence, the ice permittivity, and the mixing ratio. The function
returns the brightness temperature for the horizontal and vertical polarizations. 
Inputs:
    - theta: incidence angle
    - dice: ice thickness
    - tice: ice temperature
    - sice: ice salinity
    - snow_presence: snow presence
    - ice_permittivity: ice permittivity
    - mixing_ratio: mixing ratio
Outputs:
    - TB_RT_H: brightness temperature for the horizontal polarization
    - TB_RT_V: brightness temperature for the vertical polarization
"""
def burke_model(theta, dice, tice, sice, snow_presence, ice_permittivity, mixing_ratio):
    sw=33 #(in psu)
    tw=-1.8 #(in ºC)
    tsnow=tice
    dsnow=dice*0.1
    rho_snow=300
    freq=1.4e9
    theta=np.radians(theta) #(in radians)
    c=3e8
    if snow_presence == 0: #3 layers case (w/o snow)
        #Air
        EpRe_0=1.0
        EpIm_0=0.0
        diel_cnt_0=complex(EpRe_0,EpIm_0)
        n_0=1.0
        #Ice
        EpRe_1,EpIm_1=epsilon_Vant_ice(sice,tice)
        diel_cnt_1=complex(EpRe_1,EpIm_1)
        n_1=np.real(np.sqrt(diel_cnt_1))
        T1=tice+273.15
        #Water
        EpRe_2,EpIm_2=epsilon_water(freq,sw,tw)
        diel_cnt_2=complex(EpRe_2,EpIm_2)
        n_2=np.real(np.sqrt(diel_cnt_2))
        T2=tw+273.15
        #Air layer 0
        beta0=np.sqrt(0.5*(EpRe_0-np.sin(theta)**2)*(1+np.sqrt(1+(EpIm_0**2/(EpRe_0-np.sin(theta)**2)**2))))
        af0=EpIm_0/(2*beta0)
        kappa0=2*np.pi*freq/c*(beta0+1j*af0)
        #Thin ice layer 1
        theta_in_1=np.arcsin(n_0/n_1*np.sin(theta))
        beta1=np.sqrt(0.5*(EpRe_1-np.sin(theta_in_1)**2)*(1+np.sqrt(1+(EpIm_1**2/(EpRe_1-np.sin(theta_in_1)**2)**2))))
        af1=EpIm_1/(2*beta1)
        gamma1=2*np.pi*freq/c*af1 
        kappa1=2*np.pi*freq/c*(beta1+1j*af1)
        #Seawater layer 2 
        theta_in_2=np.arcsin(n_1/n_2*np.sin(theta_in_1))
        beta2=np.sqrt(0.5*(EpRe_2-np.sin(theta_in_2)**2)*(1+np.sqrt(1+(EpIm_2**2/(EpRe_2-np.sin(theta_in_2)**2)**2))))
        af2=EpIm_2/(2*beta2)
        gamma2=2*np.pi*freq/c*af2
        kappa2=2*np.pi*freq/c*(beta2+1j*af2)
        #Compute RO coefficients
        roH_1=abs((kappa1-kappa0)/(kappa1+kappa0))**2.0 
        roH_2=abs((kappa2-kappa1)/(kappa2+kappa1))**2.0
        roV_1=abs((diel_cnt_0*kappa1-diel_cnt_1*kappa0)/(diel_cnt_0*kappa1+diel_cnt_1*kappa0))**2.0
        roV_2=abs((diel_cnt_1*kappa2-diel_cnt_2*kappa1)/(diel_cnt_1*kappa2+diel_cnt_2*kappa1))**2.0
        TSky=5
        
        #Equation of the TB for 3 layers from Burke et al. 1979
        A1=(T1*(1-np.exp(-gamma1*dice))*(1+roH_2*np.exp(-gamma1*dice)))
        A2=T2
        B1=(1-roH_1)
        B2=(1-roH_2)*np.exp(-gamma1*dice)
        Tsf=A1*B1+A2*B1*B2
        TB_RT_H=TSky*roH_1+Tsf
        
        A1=(T1*(1-np.exp(-gamma1*dice))*(1+roV_2*np.exp(-gamma1*dice)))
        A2=T2
        B1=(1-roV_1)
        B2=(1-roV_2)*np.exp(-gamma1*dice)
        Tsf=A1*B1+A2*B1*B2
        TB_RT_V=TSky*roV_1+Tsf
        
        return TB_RT_H,TB_RT_V
    
    else: #4 layers case (w/snow)
        #Air
        EpRe_0=1.0
        EpIm_0=0.0
        diel_cnt_0=complex(EpRe_0,EpIm_0)
        n_0=1.0
        #Snow
        EpRe_1,EpIm_1=epsilon_snow(freq,rho_snow,tsnow)
        diel_cnt_1=complex(EpRe_1,EpIm_1)
        n_1=np.real(np.sqrt(diel_cnt_1))
        T1=tsnow+273.15
        #Ice
        EpRe_2,EpIm_2=epsilon_Vant_ice(sice,tice)
        diel_cnt_2=complex(EpRe_2,EpIm_2)
        n_2=np.real(np.sqrt(diel_cnt_2))
        T2=tice+273.15
        #Water
        EpRe_3,EpIm_3=epsilon_water(freq,tw,sw)   
        diel_cnt_3=complex(EpRe_3,EpIm_3)
        n_3=np.real(np.sqrt(diel_cnt_3))
        T3=tw+273.15
        #Equation of the TB for 4 layers from Burke et al. 1979
        #Air layer 0
        beta0=np.sqrt(0.5*(EpRe_0-np.sin(theta)**2)*(1+np.sqrt(1+(EpIm_0**2/(EpRe_0-np.sin(theta)**2)**2))))
        af0=EpIm_0/(2*beta0)
        kappa0=2*np.pi*freq/c*(beta0+1j*af0)
        #Snow layer 1
        theta_in_1=np.arcsin(n_0/n_1*np.sin(theta))
        beta1=np.sqrt(0.5*(EpRe_1-np.sin(theta_in_1)**2)*(1+np.sqrt(1+(EpIm_1**2/(EpRe_1-np.sin(theta_in_1)**2)**2))))
        af1=EpIm_1/(2*beta1)
        gamma1=2*np.pi*freq/c*af1 
        kappa1=2*np.pi*freq/c*(beta1+1j*af1)
        #Thin Ice layer 2 
        theta_in_2=np.arcsin(n_1/n_2*np.sin(theta_in_1))
        beta2=np.sqrt(0.5*(EpRe_2-np.sin(theta_in_2)**2)*(1+np.sqrt(1+(EpIm_2**2/(EpRe_2-np.sin(theta_in_2)**2)**2))))
        af2=EpIm_2/(2*beta2)
        gamma2=2*np.pi*freq/c*af2
        kappa2=2*np.pi*freq/c*(beta2+1j*af2)
        #Sea Water layer 3
        theta_in_3=np.arcsin(n_2/n_3*np.sin(theta_in_2))
        beta3=np.sqrt(0.5*(EpRe_3-np.sin(theta_in_3)**2)*(1+np.sqrt(1+(EpIm_3**2/(EpRe_3-np.sin(theta_in_3)**2)**2))))
        af3=EpIm_3/(2*beta3)
        gamma3=2*np.pi*freq/c*af3
        kappa3=2*np.pi*freq/c*(beta3+1j*af3)
        #Compute RO coefficients
        roH_1=abs((kappa1-kappa0)/(kappa1+kappa0))**2.0 
        roH_2=abs((kappa2-kappa1)/(kappa2+kappa1))**2.0
        roH_3=abs((kappa3-kappa2)/(kappa3+kappa2))**2.0
        roV_1=abs((diel_cnt_0*kappa1-diel_cnt_1*kappa0)/(diel_cnt_0*kappa1+diel_cnt_1*kappa0))**2.0
        roV_2=abs((diel_cnt_1*kappa2-diel_cnt_2*kappa1)/(diel_cnt_1*kappa2+diel_cnt_2*kappa1))**2.0
        roV_3=abs((diel_cnt_2*kappa3-diel_cnt_3*kappa2)/(diel_cnt_2*kappa3+diel_cnt_3*kappa2))**2.0
        TSky=5
 
        A1=(T1*(1-np.exp(-gamma1*dsnow))*(1+roH_2*np.exp(-gamma1*dsnow)))
        A21=(T2*(1-np.exp(-gamma2*dice))*(1+roH_3*np.exp(-gamma2*dice)))
        A22=T3
        B1=(1-roH_1)
        B2=(1-roH_2)*np.exp(-gamma1*dsnow)
        B3=(1-roH_3)*np.exp(-gamma2*dice)
        Tsf1=B1*A1
        Tsf2=A21*B1*B2
        Tsf3=A22*B1*B3*B2
        Tsf_H=Tsf1+Tsf2+Tsf3
        TB_RT_H=TSky*roH_1+Tsf_H
        
        A1=(T1*(1-np.exp(-gamma1*dsnow))*(1+roV_2*np.exp(-gamma1*dsnow)))
        A21=(T2*(1-np.exp(-gamma2*dice))*(1+roV_3*np.exp(-gamma2*dice)))
        A22=T3
        B1=(1-roV_1)
        B2=(1-roV_2)*np.exp(-gamma1*dsnow)
        B3=(1-roV_3)*np.exp(-gamma2*dice)
        Tsf1=B1*A1
        Tsf2=A21*B1*B2
        Tsf3=A22*B1*B3*B2
        Tsf_V=Tsf1+Tsf2+Tsf3
        TB_RT_V=TSky*roV_1+Tsf_V
        
        return TB_RT_H,TB_RT_V
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that inverts the Burke model to retrieve the ice thickness. The function receives the initial guess for the ice thickness, the incidence angle,
the brightness temperature, the ice temperature, the ice salinity, the snow presence, the ice permittivity, and the mixing ratio. The function returns the
retrieved ice thickness.
Inputs:
    - dice: initial guess for the ice thickness
    - theta: incidence angle
    - TB: brightness temperature
    - tice: ice temperature
    - sice: ice salinity
    - snow_presence: snow presence
    - ice_permittivity: ice permittivity
    - mixing_ratio: mixing ratio
Outputs:
    - d_retrieved: retrieved ice thickness
"""
def burke_model_dretrieval(dice, *conditions):
    theta, I, tice, sice, snow_presence, ice_permittivity, mixing_ratio = conditions
    sw=33 #(in psu)
    tw=-1.8 #(in ºC)
    tsnow=tice
    dsnow=dice*0.1
    rho_snow=300
    freq=1.4e9
    theta=np.radians(theta) #(in radians)
    c=3e8
    if snow_presence == 0: #3 layers case (w/o snow)
        #Air
        EpRe_0=1.0
        EpIm_0=0.0
        diel_cnt_0=complex(EpRe_0,EpIm_0)
        n_0=1.0
        #Ice
        EpRe_1,EpIm_1=epsilon_Vant_ice(sice,tice)
        diel_cnt_1=complex(EpRe_1,EpIm_1)
        n_1=np.real(np.sqrt(diel_cnt_1))
        T1=tice+273.15
        #Water
        EpRe_2,EpIm_2=epsilon_water(freq,sw,tw)
        diel_cnt_2=complex(EpRe_2,EpIm_2)
        n_2=np.real(np.sqrt(diel_cnt_2))
        T2=tw+273.15
        #Air layer 0
        beta0=np.sqrt(0.5*(EpRe_0-np.sin(theta)**2)*(1+np.sqrt(1+(EpIm_0**2/(EpRe_0-np.sin(theta)**2)**2))))
        af0=EpIm_0/(2*beta0)
        kappa0=2*np.pi*freq/c*(beta0+1j*af0)
        #Thin ice layer 1
        theta_in_1=np.arcsin(n_0/n_1*np.sin(theta))
        beta1=np.sqrt(0.5*(EpRe_1-np.sin(theta_in_1)**2)*(1+np.sqrt(1+(EpIm_1**2/(EpRe_1-np.sin(theta_in_1)**2)**2))))
        af1=EpIm_1/(2*beta1)
        gamma1=2*np.pi*freq/c*af1 
        kappa1=2*np.pi*freq/c*(beta1+1j*af1)
        #Seawater layer 2 
        theta_in_2=np.arcsin(n_1/n_2*np.sin(theta_in_1))
        beta2=np.sqrt(0.5*(EpRe_2-np.sin(theta_in_2)**2)*(1+np.sqrt(1+(EpIm_2**2/(EpRe_2-np.sin(theta_in_2)**2)**2))))
        af2=EpIm_2/(2*beta2)
        gamma2=2*np.pi*freq/c*af2
        kappa2=2*np.pi*freq/c*(beta2+1j*af2)
        #Compute RO coefficients
        roH_1=abs((kappa1-kappa0)/(kappa1+kappa0))**2.0 
        roH_2=abs((kappa2-kappa1)/(kappa2+kappa1))**2.0
        roV_1=abs((diel_cnt_0*kappa1-diel_cnt_1*kappa0)/(diel_cnt_0*kappa1+diel_cnt_1*kappa0))**2.0
        roV_2=abs((diel_cnt_1*kappa2-diel_cnt_2*kappa1)/(diel_cnt_1*kappa2+diel_cnt_2*kappa1))**2.0
        TSky=5
        
        #Equation of the TB for 3 layers from Burke et al. 1979
        A1=(T1*(1-np.exp(-gamma1*dice))*(1+roH_2*np.exp(-gamma1*dice)))
        A2=T2
        B1=(1-roH_1)
        B2=(1-roH_2)*np.exp(-gamma1*dice)
        Tsf=A1*B1+A2*B1*B2
        TB_RT_H=TSky*roH_1+Tsf
        
        A1=(T1*(1-np.exp(-gamma1*dice))*(1+roV_2*np.exp(-gamma1*dice)))
        A2=T2
        B1=(1-roV_1)
        B2=(1-roV_2)*np.exp(-gamma1*dice)
        Tsf=A1*B1+A2*B1*B2
        TB_RT_V=TSky*roV_1+Tsf
        
        return ((1/2) * (TB_RT_H + TB_RT_V)) - I
    
    else: #4 layers case (w/snow)
        #Air
        EpRe_0=1.0
        EpIm_0=0.0
        diel_cnt_0=complex(EpRe_0,EpIm_0)
        n_0=1.0
        #Snow
        EpRe_1,EpIm_1=epsilon_snow(freq,rho_snow,tsnow)
        diel_cnt_1=complex(EpRe_1,EpIm_1)
        n_1=np.real(np.sqrt(diel_cnt_1))
        T1=tsnow+273.15
        #Ice
        EpRe_2,EpIm_2=epsilon_Vant_ice(sice,tice)
        diel_cnt_2=complex(EpRe_2,EpIm_2)
        n_2=np.real(np.sqrt(diel_cnt_2))
        T2=tice+273.15
        #Water
        EpRe_3,EpIm_3=epsilon_water(freq,tw,sw)   
        diel_cnt_3=complex(EpRe_3,EpIm_3)
        n_3=np.real(np.sqrt(diel_cnt_3))
        T3=tw+273.15
        #Equation of the TB for 4 layers from Burke et al. 1979
        #Air layer 0
        beta0=np.sqrt(0.5*(EpRe_0-np.sin(theta)**2)*(1+np.sqrt(1+(EpIm_0**2/(EpRe_0-np.sin(theta)**2)**2))))
        af0=EpIm_0/(2*beta0)
        kappa0=2*np.pi*freq/c*(beta0+1j*af0)
        #Snow layer 1
        theta_in_1=np.arcsin(n_0/n_1*np.sin(theta))
        beta1=np.sqrt(0.5*(EpRe_1-np.sin(theta_in_1)**2)*(1+np.sqrt(1+(EpIm_1**2/(EpRe_1-np.sin(theta_in_1)**2)**2))))
        af1=EpIm_1/(2*beta1)
        gamma1=2*np.pi*freq/c*af1 
        kappa1=2*np.pi*freq/c*(beta1+1j*af1)
        #Thin Ice layer 2 
        theta_in_2=np.arcsin(n_1/n_2*np.sin(theta_in_1))
        beta2=np.sqrt(0.5*(EpRe_2-np.sin(theta_in_2)**2)*(1+np.sqrt(1+(EpIm_2**2/(EpRe_2-np.sin(theta_in_2)**2)**2))))
        af2=EpIm_2/(2*beta2)
        gamma2=2*np.pi*freq/c*af2
        kappa2=2*np.pi*freq/c*(beta2+1j*af2)
        #Sea Water layer 3
        theta_in_3=np.arcsin(n_2/n_3*np.sin(theta_in_2))
        beta3=np.sqrt(0.5*(EpRe_3-np.sin(theta_in_3)**2)*(1+np.sqrt(1+(EpIm_3**2/(EpRe_3-np.sin(theta_in_3)**2)**2))))
        af3=EpIm_3/(2*beta3)
        gamma3=2*np.pi*freq/c*af3
        kappa3=2*np.pi*freq/c*(beta3+1j*af3)
        #Compute RO coefficients
        roH_1=abs((kappa1-kappa0)/(kappa1+kappa0))**2.0 
        roH_2=abs((kappa2-kappa1)/(kappa2+kappa1))**2.0
        roH_3=abs((kappa3-kappa2)/(kappa3+kappa2))**2.0
        roV_1=abs((diel_cnt_0*kappa1-diel_cnt_1*kappa0)/(diel_cnt_0*kappa1+diel_cnt_1*kappa0))**2.0
        roV_2=abs((diel_cnt_1*kappa2-diel_cnt_2*kappa1)/(diel_cnt_1*kappa2+diel_cnt_2*kappa1))**2.0
        roV_3=abs((diel_cnt_2*kappa3-diel_cnt_3*kappa2)/(diel_cnt_2*kappa3+diel_cnt_3*kappa2))**2.0
        TSky=5
 
        A1=(T1*(1-np.exp(-gamma1*dsnow))*(1+roH_2*np.exp(-gamma1*dsnow)))
        A21=(T2*(1-np.exp(-gamma2*dice))*(1+roH_3*np.exp(-gamma2*dice)))
        A22=T3
        B1=(1-roH_1)
        B2=(1-roH_2)*np.exp(-gamma1*dsnow)
        B3=(1-roH_3)*np.exp(-gamma2*dice)
        Tsf1=B1*A1
        Tsf2=A21*B1*B2
        Tsf3=A22*B1*B3*B2
        Tsf_H=Tsf1+Tsf2+Tsf3
        TB_RT_H=TSky*roH_1+Tsf_H
        
        A1=(T1*(1-np.exp(-gamma1*dsnow))*(1+roV_2*np.exp(-gamma1*dsnow)))
        A21=(T2*(1-np.exp(-gamma2*dice))*(1+roV_3*np.exp(-gamma2*dice)))
        A22=T3
        B1=(1-roV_1)
        B2=(1-roV_2)*np.exp(-gamma1*dsnow)
        B3=(1-roV_3)*np.exp(-gamma2*dice)
        Tsf1=B1*A1
        Tsf2=A21*B1*B2
        Tsf3=A22*B1*B3*B2
        Tsf_V=Tsf1+Tsf2+Tsf3
        TB_RT_V=TSky*roV_1+Tsf_V
        
        return ((1/2) * (TB_RT_H + TB_RT_V)) - I
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that retrieves the ice thickness using the Burke model. The function receives the initial guess for the ice thickness, the incidence angle,
the brightness temperature, the ice temperature, the ice salinity, the snow presence, the ice permittivity, and the mixing ratio. The function returns the
retrieved ice thickness.
Inputs:
    - d0: initial guess for the ice thickness
    - I: incidence angle
    - theta: brightness temperature
    - tice: ice temperature
    - sice: ice salinity
    - snow_presence: snow presence
    - ice_permittivity: ice permittivity
    - mixing_ratio: mixing ratio
Outputs:
    - sol: retrieved ice thickness
"""
def d_retrieved_burke_solver(d0, I, theta, tice, sice, snow_presence, ice_permittivity, mixing_ratio):
    conditions = (theta, I, tice, sice, snow_presence, ice_permittivity, mixing_ratio)
    solve = root(burke_model_dretrieval, d0, args=conditions, method='lm', options={'ftol':10e-3})
    sol = solve.x[0]
    if sol > 1:
        dice = np.arange(1.01, 3.01, 0.01)
        tbh, tbv = burke_model(theta, dice, tice, sice, snow_presence, ice_permittivity, mixing_ratio)
        tb = (tbh + tbv) / 2
        dtb = np.diff(tb)
        indices = np.where(np.abs(dtb) < 0.1)[0]
        if indices.size > 0:
            dmax = dice[indices[0]]
        else:
            dmax = 1.5
        if sol > dmax:
            sol = dmax
    if sol < 0:
        sol = 0
    return sol 
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#