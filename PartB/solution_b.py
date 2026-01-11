"""
Μέρος Β: Εκτίμηση Πυκνότητας με Παράθυρα Parzen
Αναγνώριση Προτύπων & Μηχανική Μάθηση - 2025-2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_data(filepath):
    """
    Φορτώνει μονοδιάστατα δεδομένα από αρχείο CSV.
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data.flatten()  # Διασφάλιση ότι είναι 1D
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση: {e}")
        return None

def hypercube_kernel(u):
    """
    Πυρήνας υπερκύβου (ομοιόμορφος).
    K(u) = 1.0 αν |u| <= 0.5, αλλιώς 0
    (Ορισμός με βάση το μήκος πλευράς h, όχι την ακτίνα)
    """
    return np.where(np.abs(u) <= 0.5, 1.0, 0.0)

def gaussian_kernel(u):
    """
    Πυρήνας Gauss.
    K(u) = (1/sqrt(2π)) * exp(-0.5 * u^2)
    """
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def parzen_window_estimate(x, data, h, kernel_func):
    """
    Υπολογίζει την εκτίμηση Parzen Window στο σημείο x.
    Τύπος: p(x) = (1 / (N * h)) * Σ K((x - x_i) / h)
    
    Παράμετροι:
        x: σημείο(α) αξιολόγησης
        data: δεδομένα εκπαίδευσης
        h: πλάτος παραθύρου (bandwidth)
        kernel_func: συνάρτηση πυρήνα
    """
    N = len(data)
    
    if np.isscalar(x):
        x = np.array([x])
    
    # Αναδιαμόρφωση για broadcasting
    x_col = x[:, np.newaxis]      # (M, 1)
    data_row = data[np.newaxis, :] # (1, N)
    
    # Υπολογισμός κλιμακωμένων αποστάσεων
    u = (x_col - data_row) / h
    
    # Εφαρμογή πυρήνα
    k_values = kernel_func(u)  # (M, N)
    
    # Άθροισμα και κανονικοποίηση
    p_x = np.sum(k_values, axis=1) / (N * h)
    
    return p_x

def true_pdf(x):
    """
    Πραγματική κατανομή: N(1, 4) -> μέσος=1, διακύμανση=4 -> τ.α.=2
    """
    return norm.pdf(x, loc=1, scale=2)

def compute_squared_error(data, h, kernel_func):
    """
    Υπολογίζει το τετραγωνικό σφάλμα μεταξύ εκτιμηθείσας και πραγματικής PDF.
    """
    # Εκτίμηση PDF στα σημεία των δεδομένων
    p_estimated = parzen_window_estimate(data, data, h, kernel_func)
    
    # Πραγματική PDF στα ίδια σημεία
    p_true = true_pdf(data)
    
    # Άθροισμα τετραγωνικών σφαλμάτων
    error = np.sum((p_estimated - p_true)**2)
    return error

def main():
    print("--- Μέρος Β: Εκτίμηση Πυκνότητας με Παράθυρα Parzen ---")
    
    # 1. Φόρτωση δεδομένων
    filepath = '../Datasets/dataset2.csv'
    data = load_data(filepath)
    
    if data is None:
        return
        
    print(f"Φορτώθηκαν δεδομένα. Μέγεθος: {data.shape}")
    print(f"Μέσος δείγματος: {np.mean(data):.4f}, Διακύμανση: {np.var(data):.4f}")
    
    # 2. Ιστόγραμμα vs Πραγματική Κατανομή
    print("\nΔημιουργία ιστογράμματος...")
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, density=True, alpha=0.6, color='gray', label='Ιστόγραμμα Δεδομένων')
    
    x_range = np.linspace(min(data)-2, max(data)+2, 200)
    plt.plot(x_range, true_pdf(x_range), 'r-', linewidth=2, label='Πραγματική N(1, 4)')
    
    plt.title('Ιστόγραμμα vs Πραγματική Κατανομή')
    plt.xlabel('x')
    plt.ylabel('Πυκνότητα')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('histogram_verification.png')
    print("Αποθηκεύτηκε το histogram_verification.png")
    
    # 3. Εύρεση βέλτιστου h για κάθε πυρήνα
    h_values = np.arange(0.1, 10.1, 0.1)  # [0.1, 0.2, ..., 10.0]
    
    errors_hypercube = []
    errors_gaussian = []
    
    print("\nΥπολογισμός σφαλμάτων για h στο [0.1, 10]...")
    
    for h in h_values:
        # Πυρήνας υπερκύβου
        err_h = compute_squared_error(data, h, hypercube_kernel)
        errors_hypercube.append(err_h)
        
        # Πυρήνας Gauss
        err_g = compute_squared_error(data, h, gaussian_kernel)
        errors_gaussian.append(err_g)
        
    # Εύρεση βέλτιστου h
    best_h_idx_hyper = np.argmin(errors_hypercube)
    best_h_hyper = h_values[best_h_idx_hyper]
    min_err_hyper = errors_hypercube[best_h_idx_hyper]
    
    best_h_idx_gauss = np.argmin(errors_gaussian)
    best_h_gauss = h_values[best_h_idx_gauss]
    min_err_gauss = errors_gaussian[best_h_idx_gauss]
    
    print(f"\nΠυρήνας Υπερκύβου:")
    print(f"  Βέλτιστο h: {best_h_hyper:.1f}")
    print(f"  Ελάχιστο σφάλμα: {min_err_hyper:.4f}")
    
    print(f"\nΠυρήνας Gauss:")
    print(f"  Βέλτιστο h: {best_h_gauss:.1f}")
    print(f"  Ελάχιστο σφάλμα: {min_err_gauss:.4f}")
    
    # 4. Γραφήματα Σφάλματος vs h
    print("\nΔημιουργία γραφημάτων σφάλματος...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Γράφημα Υπερκύβου
    ax1.plot(h_values, errors_hypercube, 'b-')
    ax1.plot(best_h_hyper, min_err_hyper, 'ro', label=f'Βέλτιστο h={best_h_hyper:.1f}')
    ax1.set_title('Πυρήνας Υπερκύβου: Σφάλμα vs h')
    ax1.set_xlabel('h')
    ax1.set_ylabel('Τετραγωνικό Σφάλμα')
    ax1.legend()
    ax1.grid(True)
    
    # Γράφημα Gauss
    ax2.plot(h_values, errors_gaussian, 'g-')
    ax2.plot(best_h_gauss, min_err_gauss, 'ro', label=f'Βέλτιστο h={best_h_gauss:.1f}')
    ax2.set_title('Πυρήνας Gauss: Σφάλμα vs h')
    ax2.set_xlabel('h')
    ax2.set_ylabel('Τετραγωνικό Σφάλμα')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('parzen_error_plots.png')
    print("Αποθηκεύτηκε το parzen_error_plots.png")

if __name__ == "__main__":
    main()
