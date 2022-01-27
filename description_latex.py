& Name  & Keyword & Description & Source\\
\hline
  1 & \emph{LASCO start}  & LASCO\_Start & First CME appearance in LASCO C2/C3 coronographs & LASCO/CDAW\\
  2 & \emph{Start date}  & Start\_Date & Start time of CME extrapolated at 20 $R_{\odot}$ & This work\\
  3 & \emph{Arrival date}  & Arrival\_Date & Estimated arrival time of the ICME based primarily on plasma and magnetic field observations & R\&C\\
  4 & \emph{Plasma event dur.}  & PE\_duration & End of the ICME plasma signatures after col. 3 is recorded & R\&C\\
  5 & \emph{Arrival speed}  & Arrival\_v & ($km/s$) ICME arrival speed measured at L1 ($\sim$ 1AU) & R\&C\\
  6 & \emph{Transit time}  & Transit\_time & ($hrs.$) Computed between col. 3 and col. 1 & This work\\
  7 & \emph{Trans. time error}  & Transit\_time\_err & ($hrs.$) Error associated to the extrapolated start date (col. 3) of a CME & This work\\
  8 & \emph{LASCO date}  & LASCO\_Date & Most likely CME associated with the ICME observed by LASCO & LASCO/CDAW\\
  9 & \emph{LASCO speed}  & LASCO\_v & ($km/s$) Max. plane-of-sky (POS) CME speed along the angular width  & LASCO/CDAW\\
  10 & \emph{Position angle}  & LASCO\_pa & ($deg.$) Counterclockwise (from solar North) angle of appearance into coronographs & LASCO/CDAW\\
  11 & \emph{Angular width}  & LASCO\_da & ($deg.$) Angular expansion of CME into coronographs & LASCO/CDAW\\
  12 & \emph{Halo}  & LASCO\_halo & If col. 15 is $>$ \ang{270} then 'FH' (full halo), if $>$ \ang{180} 'HH' (half halo), if $>$ \ang{90} 'PH' (partial halo), otherwise 'N'. & LASCO/CDAW\\
  13 & \emph{De-proj. speed}  & v\_r & ($km/s$) De-projected CME speed (from 9, see Appendix \ref{AppendixA1}) & This work \\
  14 & \emph{De-proj. speed error}  & v\_r\_err & ($km/s$) Uncertainty of CME initial speed (col. 13) & This work\\
  15 & \emph{Theta source} & Theta & ($arcsec$) Longitude of the most likely source of CME& This work\\
  16 & \emph{Phi source} & Phi & ($arcsec$) Co-latitude of the most likely source of CME & This work\\
  17 & \emph{Source pos. error} & POS\_source\_err & ($deg.$) Uncertainty of the most likely CME source & This work\\
  18 & \emph{POS source angle} & POS\_source & ($deg.$) Principal angle of the most likely CME source & This work\\ 
  19 & \emph{Relative width} & rel\_wid & ($rad.$) De-projected width of CME & This work\\
  20 & \emph{Mass}  & Mass & ($g$) Estimated CME Mass (if provided) & LASCO/CDAW\\
  21 & \emph{Solar wind type}  & SW\_type & Solar wind (slow, S, or fast, F) interacting with the ICME & This work\\
  22 & \emph{Bz}  & Bz &($nT$) $z$-component of magnetic field at L1 and CME arrival time (col. 3) & R\&C\\
  23 & \emph{Dst}  & DST & Geomagnetic Dst index recorded at CME arrival (col. 3) & R\&C\\
  24 & \emph{Stat. de-proj. speed}  & v\_r\_stat & ($km/s$) Statistical de-projected CME speed, i.e. v\_r\_stat=LASCO\_v*1.027+41.5 & Paouris et al.\\
  25 & \emph{Acceleration} & Accel & ($m/s^2$) Residual acceleration at last CME observation & This work\\
  26 & \emph{Analytic sol. wind} & Analytic\_w & ($km/s$) solar wind from DBM exact inversion & This work\\
  27 & \emph{Analytic gamma} & Analytic\_gamma & ($km^{-1}$) drag parameter, $\gamma$, from DBM exact inversion & This work\\