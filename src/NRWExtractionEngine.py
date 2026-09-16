import numpy as np
import matplotlib.pyplot as plt

class NRWExtractionEngine:
    def __init__(self, f_start=8.0e9, f_stop=18.0e9, num_points=501, thickness=0.003175):
        """
        Initializes the Nicolson-Ross-Weir Extraction Engine parameters.
        Default thickness: 1/8 inch converted to meters (3.175 mm)
        """
        self.freqs = np.linspace(f_start, f_stop, num_points)
        self.omega = 2 * np.pi * self.freqs
        self.c = 299792458                  # Speed of light in vacuum (m/s)
        self.Z_0 = 1.0                      # Normalized free-space impedance
        self.d = thickness                  # Sample thickness (m)
        self.k_0 = self.omega / self.c
        
        # Target Material Properties (Standard FiberGlass Baseline)
        self.eps_r_true = 4.4 - 1j * (4.4 * 0.02)

    def synthesize_s_parameters(self, noise_floor=1e-4, seed=42):
        """
        Synthesizes 2-port Scattering Matrix inputs with injected bench trace noise.
        """
        k_m = self.k_0 * np.sqrt(self.eps_r_true)
        Z_m = 1.0 / np.sqrt(self.eps_r_true)
        
        gamma_interface = (Z_m - self.Z_0) / (Z_m + self.Z_0)
        P_true = np.exp(-1j * k_m * self.d)
        
        s11_clean = (gamma_interface * (1.0 - P_true**2)) / (1.0 - (gamma_interface**2) * (P_true**2))
        s21_clean = (P_true * (1.0 - gamma_interface**2)) / (1.0 - (gamma_interface**2) * (P_true**2))
        
        np.random.seed(seed)
        s11_measured = s11_clean + (np.random.normal(0, noise_floor, len(self.freqs)) + 1j * np.random.normal(0, noise_floor, len(self.freqs)))
        s21_measured = s21_clean + (np.random.normal(0, noise_floor, len(self.freqs)) + 1j * np.random.normal(0, noise_floor, len(self.freqs)))
        
        return s11_measured, s21_measured

    def execute_inversion_pipeline(self, s11_measured, s21_measured):
        """
        Executes the mathematical core of the NRW parameter inversion matrix.
        """
        # Step 1: Compute intermediate parameter X
        X = (1.0 - s21_measured**2 + s11_measured**2) / (2.0 * s11_measured)
        
        # Step 2: Extract reflection coefficient Gamma (magnitude constraint <= 1.0)
        Gamma1 = X + np.sqrt(X**2 - 1.0 + 0j)
        Gamma2 = X - np.sqrt(X**2 - 1.0 + 0j)
        Gamma = np.where(np.abs(Gamma1) <= 1.0, Gamma1, Gamma2)
        
        # Step 3: Compute the transmission propagation factor P
        P = (s11_measured + s21_measured - Gamma) / (1.0 - (s11_measured + s21_measured) * Gamma)
        
        # Step 4: Convert propagation factor to complex relative permittivity
        ln_P = np.log(P)
        phase_unwrapped = np.unwrap(np.imag(ln_P))
        ln_P_corrected = np.real(ln_P) + 1j * phase_unwrapped
        
        eps_r_extracted = (-(ln_P_corrected / (self.k_0 * self.d)) ** 2)
        epsilon_prime = np.real(eps_r_extracted)
        epsilon_double_prime = -np.imag(eps_r_extracted)
        
        # Step 5: Derive absolute Dielectric Loss Tangent
        tan_delta = epsilon_double_prime / epsilon_prime
        
        return epsilon_prime, tan_delta

    def generate_plots(self, epsilon_prime, tan_delta):
        """
        Generates and saves the metrology verification plot.
        """
        plt.close()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        
        # Top Plot: Real Permittivity
        ax1.plot(self.freqs / 1e9, epsilon_prime, color="red", label=r"Extracted $\epsilon'_r$")
        ax1.axhline(4.4, color="black", linestyle="--", label="Fiberglass Baseline (4.4)")
        ax1.set_ylabel(r"Real Permittivity ($\epsilon'_r$)")
        ax1.set_title("Nicolson-Ross-Weir Material Parameter Extraction")
        ax1.grid(True, linestyle=":")
        ax1.legend(loc="upper right")
        ax1.set_ylim(4.0, 4.8)
        
        # Bottom Plot: Loss Tangent Verification
        ax2.plot(self.freqs / 1e9, tan_delta, color="blue", label=r"Extracted $\tan \delta$")
        ax2.axhline(0.02, color="black", linestyle="--", label="Target Specification (0.02)")
        ax2.set_xlabel("Frequency (GHz)")
        ax2.set_ylabel(r"Loss Tangent ($\tan \delta$)")
        ax2.grid(True, linestyle=":")
        ax2.legend(loc="upper right")
        ax2.set_ylim(0.0, 0.04)
        
        plt.tight_layout()
        plt.savefig("plots/material_extraction_plot.png", dpi=300)
        print("[SUCCESS] Metrology verification plot exported to plots/material_extraction_plot.png")

if __name__ == "__main__":
    # Execute the structured engine pipeline sequentially
    engine = NRWExtractionEngine()
    s11, s21 = engine.synthesize_s_parameters()
    eps_prime, loss_tangent = engine.execute_inversion_pipeline(s11, s21)
    engine.generate_plots(eps_prime, loss_tangent)
