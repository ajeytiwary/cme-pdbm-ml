

description_dict = {
'CME_num': '("integer") progressive no. of CME',
'LASCO_Start': '(datetime) Date of first appearance of CME in LASCOC2 coronograph',
'Start_Date': '(datetime) Date of CME transit at 20 solar radii',
'Arrival_Date': '(datetime) Date of Plasma arrival at L1(ACE)',
'PE_duration': '(hrs) duration of plasma disturbance at L1',
'Arrival_v':' (km/s) CME arrival speed at L1',
'Transit_time': "(hrs) date difference in hours between 'Start_Date' and 'Arrival_Date'",
'Transit_time_err': '(hrs) error on the CME transit time between 20 solar radii and 1AU',
'LASCO_Date': '(datetime) CME start date as reported in the LASCO/CDAW catalog',
'LASC0_v': '(km/s) linear speed of CMEs fastest component',
'LASCO_pa': '(deg) CME principal angle, counterclockwise from North',
'LASCO_da': '(deg) CME angular width',
# 'LASCO_halo':" (string) 'FH' if LASCO_da > 270, 'HH' if LASCO_da > 180, 'PH' if LASC_da > 90, 'N' otherwise",
'v_r': '(km/s) radially de-projected CME speed',
'v_r_err':' (km/s) uncertainty of the de-projected speed',
'Theta_source': '(arcsec) longitude of the most probable source of the CME',
'Phi_source':' (arcsec) co-latitude of the most probable source of the CME',
'source_err':' (arcsec) uncertainty of the CME source on the solar disk',
'POS_source_angle':' (deg) plane-of-sky angle of the source region of the CME',
'rel_wid': '(rad) CME width relative to its lift-off speed and position',
'Mass':' (g) Mass estimation of the CME',
'Bz': '(nT) z-component of the magnetic field at L1',
'DST': '(signed int) minimum DST during CME duration at L1',
'v_r_stat':' (km/s) statistical de-projection of the CME linear speed, i.e. v_r_stat = LASCO_v*1.027+41.5 (from Pauris et al. 2020, Space Weather)',
'Accel':' (m/s ^ 2) CME acceleration between the lift-off and 20 solar radii',
'Analityc_w':' (km/s) solar wind value obtained via analytic inversion of the Drag-Based Model equations',
'Analityc_gamma': '(km ^ -1) value of the drag parameter obtained via analytic inversion of the Drag-Based Model equations',
'filename': '(string) name of the file containing PDBM information from the statistical inversion procedure'}






                # html.Div(
                #     [
                #         html.H5("Parameters"),
                #         html.P("Inital Mass (kg):", id="m0Out"),
                #         html.P("Max Thrust (N):", id="maThrustOut"),
                #         html.P("ISP (s):", id="ispOut"),
                #         html.P("C3 (rad/s):", id="c3Out"),
                #         html.P("Gravity (N):", id="gravOut"),
                #     ]
                # )