import numpy as np # type: ignore
from scipy.integrate import odeint # type: ignore

#Below describes the Fbio model

def Fbio(name,Cl_int_lumen1,Cl_int_lumen2,Cl_int_lumen3,Cl_int_liver,Cl_int_wall,logKow,MW,Clint_type,Papp_caco2,time):
    
    # #########################################
    # Anthropometric and physiological data
    # #########################################
    
    #parameters for the hypothetical average individual (population central tendency) 
    BW = 70 # in kg
    R = 1.75 # in cm
    k_transit = 1 / 4  * 3 # small intestine transit time = 4hr; 3 segments in the modeled small intestine
    Q_cardiacc = 13.88 * BW ** 0.75 # L / h / kg BW
    Q_portal = 0.2054 * Q_cardiacc
    Q_liver = 0.05359  * Q_cardiacc
    Q_rest = 0.7221 * Q_cardiacc
    V_wall = 0.0089 * BW
    V_liver = 0.0245 *  BW
    V_rest = 0.788 * BW
    M_liver =  1800 # in g; Table S1
    M_wall = 623 # in g; Table S1
    Q_gfr = 0.30998 * BW ** 0.75
    F_reabsorption = (1 - 0.06 / Q_gfr) * Papp_caco2 ** 2.7 /(34.4E-6 ** 2.7 +  Papp_caco2 ** 2.7) # Renal reabsorption efficiency; Taken from DOI: 10.1016 / j.ejps.2016.03.018

    # #########################################
    # Partitioning
    # #########################################

    #Calculating partition coefficients between water and body compositions (protein, lipid, etc)
    Klipidw = 10  ** (logKow)
    Kproteinw = 10  ** (0.73 * logKow - 0.39)
    Kplipidw = 10  ** (1.01 * logKow + 0.12)
    Kalbuminw = 10  ** (0.73 * logKow - 0.39)

    #Calculating partition coefficients between tissues and plasma
    #liver compostion 
    v_water_liver = 0.73
    v_nlipid_liver = 0.019
    v_phospholipid_liver = 0.046
    v_albumin_liver = 0.0019
    v_protein_liver = 0.17
    #plasma compostion
    v_water_plasma = 0.96
    v_nlipid_plasma = 0.0015
    v_plipid_plasma = 0.0008
    v_alnumin_plasma = 0.029
    v_protein_plasma = 0.015
    #gut wall compostion
    v_water_wall = 0.7969
    v_nlipid_wall = 0.04347
    v_plipid_wall = 0.01953
    v_albumin_wall = 0.001056
    v_protein_wall = 0.135

    Kliverwater = v_water_liver + v_nlipid_liver * Klipidw + v_phospholipid_liver * Kplipidw + v_albumin_liver * Kalbuminw + v_protein_liver * Kproteinw
    Kplasmawater = v_water_plasma + v_nlipid_plasma * Klipidw + v_plipid_plasma * Kplipidw + v_alnumin_plasma * Kalbuminw + v_protein_plasma * Kproteinw
    Kwallwater = v_water_wall + v_nlipid_wall * Klipidw + v_plipid_wall * Kplipidw + v_albumin_wall * Kalbuminw + v_protein_wall * Kproteinw
    
    K_liver2plasma = Kliverwater / Kplasmawater
    K_wall2plasma = Kwallwater / Kplasmawater

    # Calculating Krest2plasma
    # K_muscle2plasma
    v_water_muscle = 0.788
    v_nlipid_muscle = 0.0043
    v_plipid_muscle = 0.0045
    v_albumin_muscle = 0.0013
    v_protein_muscle = 0.17
    Kmusclewater = v_water_muscle + v_nlipid_muscle * Klipidw + v_plipid_muscle * Kplipidw + v_albumin_muscle * Kalbuminw + v_protein_muscle * Kproteinw
    K_muscle2plasma = Kmusclewater / Kplasmawater
    
    #Klung2plasma
    v_water_lung = 0.84
    v_nlipid_lung = 0.0102
    v_plipid_lung = 0.0098
    v_albumin_lung = 0.0054
    v_protein_lung = 0.055
    Klungwater = v_water_lung + v_nlipid_lung * Klipidw + v_plipid_lung * Kplipidw + v_albumin_lung * Kalbuminw + v_protein_lung * Kproteinw
    K_lung2plasma = Klungwater / Kplasmawater

    #Kbrain2plasma
    v_water_brain = 0.79
    v_nlipid_brain = 0.043
    v_plipid_brain = 0.067
    v_albumin_brain = 0.00004
    v_protein_brain = 0.08
    Kbrainwater = v_water_brain + v_nlipid_brain * Klipidw + v_plipid_brain * Kplipidw + v_albumin_brain * Kalbuminw + v_protein_brain * Kproteinw
    K_brain2plasma = Kbrainwater / Kplasmawater
    
    #KKidney2plasma
    v_water_kidney = 0.78
    v_nlipid_kidney = 0.012
    v_plipid_kidney = 0.035
    v_albumin_kidney = 0.0024
    v_protein_kidney = 0.16
    Kkidneywater = v_water_kidney + v_nlipid_kidney * Klipidw + v_plipid_kidney * Kplipidw + v_albumin_kidney * Kalbuminw + v_protein_kidney * Kproteinw
    K_kidney2plasma = Kkidneywater / Kkidneywater
    
    #Kblood2plasma   
    v_water_blood = 0.81
    v_nlipid_blood = 0.0013
    v_plipid_blood = 0.0022
    v_albumin_blood = 0.016
    v_protein_blood = 0.16
    Kbloodwater = v_water_blood + v_nlipid_blood * Klipidw + v_plipid_blood * Kplipidw + v_albumin_blood * Kalbuminw + v_protein_blood * Kproteinw
    K_blood2plasma = Kbloodwater / Kplasmawater
    
    # Krest2plasma
    K_rest2plasma = (0.3842 * K_muscle2plasma + 0.007235 * K_lung2plasma + 0.019310 * K_brain2plasma + 0.00419 * K_kidney2plasma)/(0.3842 + 0.007235 + 0.019310 + 0.00419)
    
    # calculating the fraction unbound to plasma and Rblood2plasma 
    #fup = (1 / Kplasmawater) * v_water_plasma
    fup = (1 / Kbloodwater) * v_water_blood
    R_blood2plasma = K_blood2plasma
        
    # #########################################
    # Metabolism: In vitro - in vivo extrapolation (IVIVE)
    # #########################################
    
    # Absorption within the gut lumen
    Peff = 10 ** (0.2940 * np.log10(Papp_caco2) - 2.4209)  #SI Equation 18
    k_abs = (2 * Peff / R) * 3600 #SI Equation 17

    # Metablism within the gut lumen
    fub_lumen = 1 #Table S1
    V_content = 260 #Table S1

    if name == "DEHP":
        N_lumen = 3 #Table S1
    else: N_lumen = 1 #Table S1

    k_lumen1 = Cl_int_lumen1 * 0.001 * 60 * N_lumen * V_content / V_content
    k_lumen2 = Cl_int_lumen2 * 0.001 * 60 * N_lumen * V_content / V_content
    k_lumen3 = Cl_int_lumen3 * 0.001 * 60 * N_lumen * V_content / V_content
            
    #Metabolism within the liver
    w_assay_plasma = 1 #SI Table S1
    if Clint_type == "microsome": #microsomes based in vitro assay
        N_liver = 62.9 #Table S1
        v_albumin_assay = 0
        v_nlipid_assay = 0
        v_protein_assay = 1 / 1000
        v_plipid_assay = 0.35 * v_protein_assay
        v_water_assay = 1 - v_protein_assay - v_plipid_assay - v_albumin_assay - v_nlipid_assay
        Kassaywater = v_water_assay + v_nlipid_assay * Klipidw + v_plipid_assay * Kplipidw + v_albumin_assay * Kalbuminw + v_protein_assay * Kproteinw
        fu_mic = (1 / Kassaywater) * v_water_assay
        Cl_invivo_liver = Cl_int_liver * N_liver * M_liver * 60 / 1E6 * w_assay_plasma / fu_mic
        HL_invivo_liver = np.log(2)/(Cl_invivo_liver / V_liver)

    elif Clint_type == "hep": # hepatocyte based in vitro assay
        N_hep = 110
        Vr = 0.5
        fub_hep = 1 /(1 + 125 * Vr * 10  ** (0.072 * logKow  ** 2 + 0.067 * logKow - 1.126)) # Taken from DOI: 10.1124 / dmd.108.020834
        Cl_invivo_liver = Cl_int_liver * N_hep * M_liver * 60 / 1E6 * w_assay_plasma / fub_hep  #In unit of L / h for a 70kg human # unscaled
        HL_invivo_liver = np.log(2)/(Cl_invivo_liver / V_liver)

    else: print ("error")

    #Metabolism within the gut wall
    N_wall = 3.9 #Table S1 
    Cl_invivo_wall = Cl_int_wall * N_wall * M_wall * 60 / 1E6 * w_assay_plasma / fu_mic 
    HL_invivo_wall = np.log(2)/(Cl_invivo_wall / V_wall)    
    
    #Note that Cl_invivo_liver and CL_invivo_wall are the clearances used in PBTK models where fup explicitly appears in the model equations (e.g., the HTTK model by the U.S. EPA).  
    #They should be distingushed from Cl_liver and Cl_wall in our model
    #Our Cl_liver and Cl_wall already incoporate fub (see SI Text S1)
    Cl_liver = Cl_invivo_liver * fup  #IN unit of L / h for a 70kg human # unscaled
    Cl_wall = Cl_invivo_wall * fup

    # #########################################
    # PBTK model equations
    # #########################################

    def model(z,t):
    
        dA_lumen1 = - k_abs * z[0] - k_transit * z[0] - k_lumen1 * z[0] 
        dA_lumen2 = + k_transit * z[0] - k_transit * z[1] - k_abs * z[1] - k_lumen2 * z[1]  
        dA_lumen3 = + k_transit * z[1] - k_transit * z[2] - k_abs * z[2] - k_lumen3 * z[2]  

        dAlumen2wall = k_abs * (z[0] + z[1] + z[2]) 
        dArest2wall = Q_portal * R_blood2plasma / K_rest2plasma * z[5] 
        dAwall2liver = Q_portal * R_blood2plasma / K_wall2plasma * z[3]  
        dArest2liver = Q_liver * R_blood2plasma / K_rest2plasma * z[5] 
        dAliver2rest = (Q_liver + Q_portal) * R_blood2plasma / K_liver2plasma * z[4]
        dAwall2met = Cl_wall * R_blood2plasma / K_wall2plasma * z[3]
        dAliver2met =  Cl_liver * R_blood2plasma / K_liver2plasma * z[4]
        dArest2urine = Q_gfr * fup * (1 - F_reabsorption)/ K_rest2plasma * z[5]
        
        dC_wall = (dAlumen2wall + dArest2wall - dAwall2liver - dAwall2met)/ V_wall
        dC_liver = (dAwall2liver + dArest2liver - dAliver2rest - dAliver2met)/ V_liver
        dC_rest = (dAliver2rest - dArest2liver - dArest2wall -  dArest2urine)/ V_rest
        
        dzdt = [dA_lumen1,dA_lumen2,dA_lumen3,dC_wall,dC_liver,dC_rest,dAlumen2wall,dArest2wall,dAwall2liver,dArest2liver,dAliver2rest]
        
        return dzdt

    # Initial condition
    z0 = [1,0,0,0,0,0,0,0,0,0,0] 

    # 10000 steps in integration
    t = np.linspace(0,time,10000)

    # Solve ODE
    z = odeint(model,z0,t)
    
    # return Fbio value
    return((z[-1,6] /(z[-1,6] + z[-1,7])) * (z[-1,8] /(z[-1,8] + z[-1,9])) * z[-1,10])
 

# Below are input parameters describing a modeled chemical
# Users can replace the following values with those for a chemical of interest.
name = "DEHP"   #Chemical name
Cl_int_lumen1 = 10.4 #Table S2
Cl_int_lumen2 = 15.6 #Table S2
Cl_int_lumen3 = 15.6 #Table S2
Cl_int_liver = 30.1 #Table S2
Cl_int_wall = 219.6 #Table S2
logKow = 7.43 #Table S2
MW = 390.6 #Table S2
Clint_type = "microsome"
Papp_caco2 = 2.1 * 10 ** -6 #Table S2
time = 24 # 24hr after ingestion; as specified in the paper

DEHP_Fbio = Fbio(name,Cl_int_lumen1,Cl_int_lumen2,Cl_int_lumen3,Cl_int_liver,Cl_int_wall,logKow,MW,Clint_type,Papp_caco2,time)
print("The theoretical bioavailablity of DEHP after 24 hr is", DEHP_Fbio)
