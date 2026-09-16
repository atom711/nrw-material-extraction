import numpy as np
import matplotlib.pyplot as plt
plt.close()

# =====================================================================
# SYSTEM PARAMETERS & CONFIGURATION
# =====================================================================
# Target: 8 to 18 GHz operational radar bandwidth
f_start = 8.0e9  # 8 GHz
f_stop = 18.0e9  # 18 GHz
num_points = 501

freqs = np.linspace(f_start, f_stop, num_points)
omega = 2 * np.pi * freqs
c = 299792458  # Speed of light in vacuum (m/s)

# Physical test sample dimensions
d = 0.003175  # Sample thickness: 1/8 inch converted to meters (3.175 mm)

# =====================================================================
# SYNTHESIZE S-PARAMETER INPUTS (Simulating VNA Ingestion)
# Baseline Target: Epsilon_prime = 4.4, Loss Tangent = 0.02 (Fiberglass)
# =====================================================================
eps_r_true = 4.4 - 1j * (4.4 * 0.02)  # Complex relative permittivity
k_0 = omega / c
k_m = k_0 * np.sqrt(eps_r_true)  # Wavevector inside the material

# Standard wave impedance transformations for normal incidence
Z_0 = 1.0  # Normalized free-space impedance
Z_m = 1.0 / np.sqrt(eps_r_true)  # Material impedance

# Interfacial reflection coefficient (air to substrate)
gamma_interface = (Z_m - Z_0) / (Z_m + Z_0)

# Internal propagation factor across sample thickness d
P_true = np.exp(-1j * k_m * d)

# Analytical Scattering Matrix generation (including infinite internal reflections)
s11_clean = (gamma_interface * (1.0 - P_true**2)) / (
    1.0 - (gamma_interface**2) * (P_true**2)
)
s21_clean = (P_true * (1.0 - gamma_interface**2)) / (
    1.0 - (gamma_interface**2) * (P_true**2)
)

# Inject synthetic bench trace noise (complex Gaussian noise floor)
np.random.seed(42)
noise_floor = 1e-4
s11_measured = s11_clean + (
    np.random.normal(0, noise_floor, num_points)
    + 1j * np.random.normal(0, noise_floor, num_points)
)
s21_measured = s21_clean + (
    np.random.normal(0, noise_floor, num_points)
    + 1j * np.random.normal(0, noise_floor, num_points)
)

# =====================================================================
# NICOLSON-ROSS-WEIR (NRW) EXTRACTION ENGINE
# =====================================================================
# Step 1: Compute intermediate parameter X
X = (1.0 - s21_measured**2 + s11_measured**2) / (2.0 * s11_measured)

# Step 2: Extract reflection coefficient Gamma (ensuring magnitude constraint <= 1.0)
Gamma1 = X + np.sqrt(X**2 - 1.0 + 0j)
Gamma2 = X - np.sqrt(X**2 - 1.0 + 0j)
Gamma = np.where(np.abs(Gamma1) <= 1.0, Gamma1, Gamma2)

# Step 3: Compute the transmission propagation factor P
P = (s11_measured + s21_measured - Gamma) / (
    1.0 - (s11_measured + s21_measured) * Gamma
)

# Step 4: Convert propagation factor to complex relative permittivity
# Inverting P = exp(-j * k_0 * sqrt(eps_r) * d) via complex natural logarithm
ln_P = np.log(P)

# Unwrapping the phase component to eliminate 2pi branch ambiguities
phase_unwrapped = np.unwrap(np.imag(ln_P))
ln_P_corrected = np.real(ln_P) + 1j * phase_unwrapped

# Solve back-calculated wavevector equality
eps_r_extracted = (-(ln_P_corrected / (k_0 * d)) ** 2)

# Isolate constitutive components of the complex vector
epsilon_prime = np.real(eps_r_extracted)
epsilon_double_prime = -np.imag(eps_r_extracted)

# Step 5: Derive the absolute Dielectric Loss Tangent
tan_delta = epsilon_double_prime / epsilon_prime

# =====================================================================
# DATA VISUALIZATION WINDOW (Inline Pane Configured)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Top Plot: Real Permittivity (Energy Storage Curve)
ax1.plot(freqs / 1e9, epsilon_prime, color="red", label=r"Extracted $\epsilon'_r$")
ax1.axhline(4.4, color="black", linestyle="--", label="Fiberglass Baseline (4.4)")
ax1.set_ylabel(r"Real Permittivity ($\epsilon'_r$)")
ax1.set_title("Nicolson-Ross-Weir Material Parameter Extraction")
ax1.grid(True, linestyle=":")
ax1.legend(loc="upper right")
ax1.set_ylim(4.0, 4.8)

# Bottom Plot: Absolute Loss Tangent Verification Window
ax2.plot(freqs / 1e9, tan_delta, color="blue", label=r"Extracted $\tan \delta$")
ax2.axhline(0.02, color="black", linestyle="--", label="Target Specification (0.02)")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel(r"Loss Tangent ($\tan \delta$)")
ax2.grid(True, linestyle=":")
ax2.legend(loc="upper right")
ax2.set_ylim(0.0, 0.04)

# Save the plot asset to your local directory before displaying it
plt.tight_layout()
plt.savefig("RF labs/schematics_graphs/lab12_material_extraction_plot.png", dpi=300)
plt.show()

