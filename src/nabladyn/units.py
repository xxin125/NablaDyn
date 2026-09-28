"""Units and physical constants.

NablaDyn works in nm, ps, Da (g/mol), kJ/mol, K and elementary charges e. Multiply a
value by a unit below to convert it into these units; divide to convert back. Numbers and
tensors both work; tensors keep their dtype, device and gradients.

    positions = positions_in_angstrom * units.ANGSTROM   # nm
    pressure_in_bar = pressure / units.BAR
    kT = units.BOLTZMANN * temperature                    # kJ/mol
"""

# Physical constants (SI 2019 exact values; CODATA 2018 for the vacuum permittivity).
AVOGADRO = 6.02214076e23  # 1/mol
BOLTZMANN = 0.00831446261815324  # kJ/(mol K): k_B N_A, the molar gas constant
COULOMB = 138.93545764438198  # kJ nm/(mol e^2): N_A e^2 / (4 pi eps0)

# Length and time.
ANGSTROM = 0.1  # nm
FS = 0.001  # ps

# Energy per mole.
KCAL_PER_MOL = 4.184  # kJ/mol, the thermochemical calorie
EV = 96.48533212331002  # kJ/mol: e x 1 V x N_A

# Pressure and mass density.
BAR = 0.0602214076  # kJ/(mol nm^3): 1e5 Pa x 1e-27 m^3/nm^3 x N_A
ATM = 1.01325 * BAR  # kJ/(mol nm^3)
G_PER_CM3 = 602.214076  # Da/nm^3: 1 g is N_A Da, and 1 cm^3 is 1e21 nm^3

__all__ = (
    "AVOGADRO",
    "BOLTZMANN",
    "COULOMB",
    "ANGSTROM",
    "FS",
    "KCAL_PER_MOL",
    "EV",
    "BAR",
    "ATM",
    "G_PER_CM3",
)
