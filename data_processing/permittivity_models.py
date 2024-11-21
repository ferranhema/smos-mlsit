#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Library imports
import numpy as np
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that calculates the permittivity of the sea ice from the Vant et al. empirical formulation.
Inputs:
    - S: salinity of the sea ice
    - T: temperature of the sea ice
Outputs:
    - EpRe: real part of the permittivity of the sea ice
    - EpIm: imaginary part of the permittivity of the sea ice
"""
def epsilon_Vant_ice(S,T):
    rho_ice = 0.917 - 0.1404*(10**-3)*T #in Mg/m^3, density of pure ice from Pounder, 1965
    # coefficients from Cox and Weeks, 1983
    if T > 0:
            a1 = -0.041221
            b1 = -18.407
            c1 = 0.58402
            d1 = 0.21454
            a2 = 0.090312
            b2 = -0.016111
            c2 = 1.2291*(10**-4)
            d2 = 1.3603*(10**-4)
    if T <= 0 and T > -2:
            a1 = -0.041221
            b1 = -18.407
            c1 = 0.58402
            d1 = 0.21454
            a2 = 0.090312
            b2 = -0.016111
            c2 = 1.2291*(10**-4)
            d2 = 1.3603*(10**-4)
    if T <= -2 and T >= -22.9:
            a1 = -4.732
            b1 = -22.45
            c1 = -0.6397
            d1 = -0.01074
            a2 = 0.08903
            b2 = -0.01763
            c2 = -5.330*(10**-4)
            d2 = -8.801*(10**-6)
    if T < -22.9:
            a1 = 9899
            b1 = 1309
            c1 = 55.27
            d1 = 0.7160
            a2 = 8.547
            b2 = 1.089
            c2 = 0.04518
            d2 = 5.819*(10**-4)                
    F1 = a1 + b1*T + c1*(T**2) + d1*(T**3)
    F2 = a2 + b2*T + c2*(T**2) + d2*(T**3)
    Vb=(rho_ice*S)/(F1-rho_ice*S*F2)
    EpRe=3.1 + 0.0084*Vb*(10**3)
    EpIm=0.037 + 0.00445*Vb*(10**3) #firstyear ice
    return EpRe,EpIm
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that computes the permittivity of the snow
Inputs:
    - freq: frequency in GHz
    - rho_snow: density of the snow in kg/m^3
    - T: temperature of the snow in K
Outputs:
    - EpRe: real part of the permittivity of the snow
    - EpIm: imaginary part of the permittivity of the snow
"""
def epsilon_snow(freq,rho_snow,T):
    T=T+273.15
    freq=freq/1e9
    if rho_snow <= 400:
        rho_snow=rho_snow/1e3
        EpRe=1+1.599*rho_snow+1.861*rho_snow**3
    else:
        v=rho_snow/917 
        e_h=1
        e_s=3.215
        EpRe=(1-v)*e_h+v*e_s
    B1=0.0207
    b=335
    B2=1.16e-11
    beta=(B1/T)*(np.exp(b/T)/np.exp(b/T)-1)**2+B2*freq**2+np.exp(-9.963+0.0372*(T-273.16))
    theta=(300/T)-1
    alpha=(0.00504+0.0062)*np.exp(-22.1*theta)
    e_i=(alpha/freq) + beta*freq    
    EpIm=e_i*(0.52*rho_snow+0.62*rho_snow**2)
    return EpRe,EpIm
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
"""
Function that computes the permittivity of the sea water
Inputs:
    - freq: frequency in Hz
    - T: temperature in ºC
    - S: salinity in PSU
Outputs:
    - EpRe: real part of the permittivity of the sea water
    - EpIm: imaginary part of the permittivity of the sea water
"""
def epsilon_water(freq,T,S):
    E0=8.854*10**(-12.0)
    Eswinf=4.9
    Esw00=87.174-1.949*10**(-1.0)*T-1.279*10**(-2.0)*(T**2.0)+2.491*10**(-4.0)*(T**3.0)
    a=1.0+1.613*10**(-5.0)*T*S-3.656*10**(-3.0)*S+3.21*10**(-5.0)*(S**2.0)-4.232*10**(-7.0)*(S**3.0)
    Esw0=Esw00*a
    Tausw0=1.1109*10**(-10.0)-3.824*10**(-12.0)*T+6.238*10**(-14.0)*(T**2.0)-5.096*10**(-16.0)*(T**3.0)
    b=1.0+2.282*10**(-5.0)*T*S-7.638*10**(-4.0)*S-7.760*10**(-6.0)*(S**2.0)+1.105*10**(-8.0)*(S**3.0)
    Tausw=(Tausw0/(2*np.pi))*b
    EpRe=Eswinf+((Esw0-Eswinf)/(1.0+((2*np.pi*freq*Tausw)**2.0)))
    si1=S*(0.18252-1.4619*10**(-3.0)*S+2.093*10**(-5.0)*(S**2.0)-1.282*10**(-7.0)*(S**3.0))
    inc=25.0-T
    phi=inc*(2.033*10**(-2.0)+1.266*10**(-4.0)*inc+2.464*10**(-6.0)*(inc**2.0)-S*(1.849*10**(-5.0)-2.551*10**(-7.0)*inc+2.551*10**(-8.0)*(inc**2.0)))
    si=si1*np.exp(-phi)
    Epimn=(2*np.pi*freq*Tausw*(Esw0-Eswinf))
    Epimd=1.0+((2*np.pi*freq*Tausw)**2.0)
    C=(si/(2*np.pi*E0*freq))
    EpIm=((Epimn/Epimd)+C)    
    return EpRe,EpIm
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#