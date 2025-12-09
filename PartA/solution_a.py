"""
Μέρος Α: Εκτίμηση Μέγιστης Πιθανοφάνειας (Maximum Likelihood Estimation)
Αναγνώριση Προτύπων & Μηχανική Μάθηση - 2025-2026
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(filepath):
    """
    Φορτώνει τα δεδομένα από αρχείο CSV.
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση: {e}")
        return None

def mle_mean(data):
    """
    Υπολογίζει την εκτίμηση μέγιστης πιθανοφάνειας του μέσου διανύσματος.
    Τύπος: μ = (1/N) * Σ(x_i)
    """
    N = data.shape[0]
    # Άθροισμα κατά μήκος του άξονα 0 (κατακόρυφα) για κάθε χαρακτηριστικό
    sum_features = np.sum(data, axis=0)
    mean_vector = sum_features / N
    return mean_vector

def mle_covariance(data, mean_vector):
    """
    Υπολογίζει την εκτίμηση μέγιστης πιθανοφάνειας του πίνακα συνδιακύμανσης.
    Τύπος: Σ = (1/N) * Σ(x_i - μ)(x_i - μ)^T
    
    Σημείωση: Χρησιμοποιούμε N (όχι N-1) καθώς αυτή είναι η MLE εκτίμηση.
    """
    N = data.shape[0]
    
    # Κεντράρισμα δεδομένων (αφαίρεση μέσου)
    centered_data = data - mean_vector
    
    # Υπολογισμός πίνακα συνδιακύμανσης με πολλαπλασιασμό πινάκων
    # (x - μ)^T * (x - μ) ισοδύναμο με centered_data.T @ centered_data
    covariance_matrix = (centered_data.T @ centered_data) / N
    
    return covariance_matrix

def gaussian_pdf_2d(x, mean, cov):
    """
    Υπολογίζει τη συνάρτηση πυκνότητας πιθανότητας 2D Gaussian στο σημείο x.
    Τύπος: p(x) = (1 / (2π|Σ|^0.5)) * exp(-0.5 * (x-μ)^T * Σ^-1 * (x-μ))
    """
    # Διασφάλιση ότι το x είναι 2D πίνακας για vectorized πράξεις
    if x.ndim == 1:
        x = x.reshape(1, -1)
        
    # Υπολογισμός ορίζουσας και αντίστροφου του πίνακα συνδιακύμανσης
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    
    # Σταθερά κανονικοποίησης
    norm_const = 1.0 / ((2 * np.pi) * np.sqrt(det_cov))
    
    # Υπολογισμός απόστασης Mahalanobis για όλα τα σημεία
    diff = x - mean
    
    # (x - μ)^T * Σ^-1 * (x - μ) υπολογιζόμενο αποδοτικά
    exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
    
    pdf = norm_const * np.exp(exponent)
    
    return pdf

def main():
    print("--- Μέρος Α: Εκτίμηση Μέγιστης Πιθανοφάνειας ---")
    
    # 1. Φόρτωση δεδομένων
    filepath = '../Datasets/dataset1.csv'
    data = load_data(filepath)
    
    if data is None:
        return

    # Διαχωρισμός χαρακτηριστικών (X) και ετικετών (y)
    X = data[:, :2]  # Πρώτες δύο στήλες: χαρακτηριστικά
    y = data[:, 2]   # Τρίτη στήλη: ετικέτες κλάσεων
    
    print(f"Φορτώθηκαν δεδομένα. Διάσταση: {X.shape}")
    
    # 2. Διαχωρισμός δεδομένων ανά κλάση
    X_c0 = X[y == 0]
    X_c1 = X[y == 1]
    X_c2 = X[y == 2]
    
    print(f"Κλάση 0: {X_c0.shape[0]} δείγματα")
    print(f"Κλάση 1: {X_c1.shape[0]} δείγματα")
    print(f"Κλάση 2: {X_c2.shape[0]} δείγματα")
    
    # 3. Υπολογισμός παραμέτρων MLE για κάθε κλάση
    # Κλάση 0
    mu_0 = mle_mean(X_c0)
    sigma_0 = mle_covariance(X_c0, mu_0)
    
    # Κλάση 1
    mu_1 = mle_mean(X_c1)
    sigma_1 = mle_covariance(X_c1, mu_1)
    
    # Κλάση 2
    mu_2 = mle_mean(X_c2)
    sigma_2 = mle_covariance(X_c2, mu_2)
    
    # Εκτύπωση αποτελεσμάτων
    print("\n--- Εκτιμηθείσες Παράμετροι ---")
    print(f"Κλάση 0 - Μέσος: {mu_0}")
    print(f"Κλάση 0 - Συνδιακύμανση:\n{sigma_0}")
    
    print(f"\nΚλάση 1 - Μέσος: {mu_1}")
    print(f"Κλάση 1 - Συνδιακύμανση:\n{sigma_1}")
    
    print(f"\nΚλάση 2 - Μέσος: {mu_2}")
    print(f"Κλάση 2 - Συνδιακύμανση:\n{sigma_2}")
    
    # Υπολογισμός και αποθήκευση μέγιστων τιμών πυκνότητας
    max_p0 = gaussian_pdf_2d(mu_0, mu_0, sigma_0)[0]
    max_p1 = gaussian_pdf_2d(mu_1, mu_1, sigma_1)[0]
    max_p2 = gaussian_pdf_2d(mu_2, mu_2, sigma_2)[0]
    
    # Υπολογισμός οριζουσών (Determinants) - επηρεάζουν το ύψος της κορυφής
    det_0 = np.linalg.det(sigma_0)
    det_1 = np.linalg.det(sigma_1)
    det_2 = np.linalg.det(sigma_2)
    
    log_text = (
        "=== ΑΝΑΛΥΤΙΚΗ ΑΝΑΦΟΡΑ ΠΑΡΑΜΕΤΡΩΝ MLE ===\n\n"
        "--- ΚΛΑΣΗ 0 ---\n"
        f"Μέσος (Mean Vector):\n{mu_0}\n"
        f"Πίνακας Συνδιακύμανσης (Covariance Matrix):\n{sigma_0}\n"
        f"Ορίζουσα (Determinant): {det_0:.4f}\n"
        f"Μέγιστη Πυκνότητα (Peak Density): {max_p0:.6f}\n\n"
        
        "--- ΚΛΑΣΗ 1 ---\n"
        f"Μέσος (Mean Vector):\n{mu_1}\n"
        f"Πίνακας Συνδιακύμανσης (Covariance Matrix):\n{sigma_1}\n"
        f"Ορίζουσα (Determinant): {det_1:.4f}\n"
        f"Μέγιστη Πυκνότητα (Peak Density): {max_p1:.6f}\n\n"
        
        "--- ΚΛΑΣΗ 2 ---\n"
        f"Μέσος (Mean Vector):\n{mu_2}\n"
        f"Πίνακας Συνδιακύμανσης (Covariance Matrix):\n{sigma_2}\n"
        f"Ορίζουσα (Determinant): {det_2:.4f}\n"
        f"Μέγιστη Πυκνότητα (Peak Density): {max_p2:.6f}\n\n"
        
        "--- ΣΥΓΚΡΙΣΗ ---\n"
        f"Η Κλάση 1 έχει τη μικρότερη ορίζουσα ({det_1:.4f}), άρα είναι η πιο 'στενή' και 'ψηλή'.\n"
        f"Η Κλάση 0 έχει τη μεγαλύτερη ορίζουσα ({det_0:.4f}), άρα είναι η πιο 'πλατιά' και 'χαμηλή'.\n"
    )
    print("\n" + log_text)
    
    with open('density_peaks.txt', 'w') as f:
        f.write(log_text)
    print("Αποθηκεύτηκε το density_peaks.txt (Εμπλουτισμένο)")
    
    # 4. Οπτικοποίηση σε 3D γράφημα
    print("\nΔημιουργία 3D γραφήματος...")
    
    # Ρύθμιση στυλ σε σκοτεινό φόντο
    plt.style.use('dark_background')
    
    # Δημιουργία πλέγματος για την απεικόνιση
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Μετατροπή πλέγματος σε σημεία για υπολογισμό PDF
    grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Υπολογισμός τιμών PDF για κάθε κλάση
    Z0 = gaussian_pdf_2d(grid_points, mu_0, sigma_0).reshape(X_grid.shape)
    Z1 = gaussian_pdf_2d(grid_points, mu_1, sigma_1).reshape(X_grid.shape)
    Z2 = gaussian_pdf_2d(grid_points, mu_2, sigma_2).reshape(X_grid.shape)
    
    # Δημιουργία 3D γραφήματος
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ορισμός σταθερού εύρους στον άξονα Z για σωστή σύγκριση
    ax.set_zlim(0, 0.025)
    
    # Προσαρμογή εμφάνισης άξονα (αφαίρεση γκρι φόντου panels)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Δημιουργία custom colormaps (Μαύρο -> Χρώμα)
    from matplotlib.colors import LinearSegmentedColormap
    
    def create_cmap(color_name, name):
        colors = [(0, 0, 0), color_name] # Black to Color
        return LinearSegmentedColormap.from_list(name, colors)
    
    cmap_red = create_cmap('red', 'BlackRed')
    cmap_green = create_cmap('lime', 'BlackGreen') # Lime is brighter than green
    cmap_blue = create_cmap('cyan', 'BlackBlue')   # Cyan pops more than blue
    
    # Σχεδίαση επιφανειών με gradient χρώματα
    # Αυξάνουμε τη διαφάνεια (alpha) για τις Κλάσεις 1 και 2, μειώνουμε για Κλάση 0
    surf0 = ax.plot_surface(X_grid, Y_grid, Z0, cmap=cmap_red, alpha=0.6, antialiased=True) # More transparent
    surf1 = ax.plot_surface(X_grid, Y_grid, Z1, cmap=cmap_green, alpha=1.0, antialiased=True) # Full opacity
    surf2 = ax.plot_surface(X_grid, Y_grid, Z2, cmap=cmap_blue, alpha=1.0, antialiased=True)  # Full opacity
    
    # Προσθήκη επίπεδου πατώματος (flat floor) στο z=0
    ax.contourf(X_grid, Y_grid, Z0, zdir='z', offset=0, cmap=cmap_red, alpha=0.3)
    ax.contourf(X_grid, Y_grid, Z1, zdir='z', offset=0, cmap=cmap_green, alpha=0.4)
    ax.contourf(X_grid, Y_grid, Z2, zdir='z', offset=0, cmap=cmap_blue, alpha=0.4)
    
    # Προσθήκη ετικετών και τίτλου
    ax.set_xlabel('Χαρακτηριστικό 1', color='white')
    ax.set_ylabel('Χαρακτηριστικό 2', color='white')
    ax.set_zlabel('Πυκνότητα Πιθανότητας', color='white')
    ax.set_title('3D Απεικόνιση Κατανομών Gauss (MLE)', color='white', fontsize=14)
    
    # Ρύθμιση γωνίας θέασης (περιστροφή δεξιόστροφα)
    ax.view_init(elev=30, azim=210)
    
    # Υπόμνημα με proxy artists
    import matplotlib.patches as mpatches
    patch0 = mpatches.Patch(color='red', label='Κλάση 0', alpha=0.6)
    patch1 = mpatches.Patch(color='lime', label='Κλάση 1', alpha=1.0)
    patch2 = mpatches.Patch(color='cyan', label='Κλάση 2', alpha=1.0)
    
    legend = ax.legend(handles=[patch0, patch1, patch2], loc='upper right')
    plt.setp(legend.get_texts(), color='white')
    
    plt.tight_layout()
    output_file = 'gaussian_3d_plot.svg'
    plt.savefig(output_file, format='svg', dpi=300, transparent=True)
    print(f"Το γράφημα αποθηκεύτηκε στο {output_file}")
    
    # ---------------------------------------------------------
    # Δημιουργία Διαδραστικού Γραφήματος (Plotly)
    # ---------------------------------------------------------
    print("\nΔημιουργία διαδραστικού γραφήματος (HTML)...")
    try:
        import plotly.graph_objects as go
        
        fig_ply = go.Figure()
        
        # Custom colorscales for Plotly (Black -> Color)
        # 0.0 is Black, 1.0 is Color
        cs_red = [[0, 'black'], [1, 'red']]
        cs_green = [[0, 'black'], [1, 'lime']]
        cs_blue = [[0, 'black'], [1, 'cyan']]
        
        # Κλάση 0 (Πιο διαφανής)
        fig_ply.add_trace(go.Surface(z=Z0, x=X_grid, y=Y_grid, 
                                   colorscale=cs_red, opacity=0.6, name='Κλάση 0', showscale=False))
        # Πάτωμα για Κλάση 0
        fig_ply.add_trace(go.Contour(z=Z0, x=x_range, y=y_range, 
                                   colorscale=cs_red, opacity=0.3, showscale=False, 
                                   contours=dict(start=0, end=0.1, size=0.01), zmin=0, zmax=0.1))
        
        # Κλάση 1 (Αυξημένη αδιαφάνεια)
        fig_ply.add_trace(go.Surface(z=Z1, x=X_grid, y=Y_grid, 
                                   colorscale=cs_green, opacity=1.0, name='Κλάση 1', showscale=False))
        # Πάτωμα για Κλάση 1
        fig_ply.add_trace(go.Contour(z=Z1, x=x_range, y=y_range, 
                                   colorscale=cs_green, opacity=0.5, showscale=False,
                                   contours=dict(start=0, end=0.1, size=0.01), zmin=0, zmax=0.1))
        
        # Κλάση 2 (Αυξημένη αδιαφάνεια)
        fig_ply.add_trace(go.Surface(z=Z2, x=X_grid, y=Y_grid, 
                                   colorscale=cs_blue, opacity=1.0, name='Κλάση 2', showscale=False))
        # Πάτωμα για Κλάση 2
        fig_ply.add_trace(go.Contour(z=Z2, x=x_range, y=y_range, 
                                   colorscale=cs_blue, opacity=0.5, showscale=False,
                                   contours=dict(start=0, end=0.1, size=0.01), zmin=0, zmax=0.1))
        
        fig_ply.update_layout(
            title='3D Απεικόνιση Κατανομών Gauss (Διαδραστικό)',
            title_font_color="white",
            paper_bgcolor="black",
            plot_bgcolor="black",
            scene=dict(
                xaxis=dict(title='Χαρακτηριστικό 1', color="white", gridcolor="#333333", tickcolor="white"),
                yaxis=dict(title='Χαρακτηριστικό 2', color="white", gridcolor="#333333", tickcolor="white"),
                zaxis=dict(title='Πυκνότητα', color="white", gridcolor="#333333", tickcolor="white", range=[0, 0.025]),
                bgcolor="black",
                camera=dict(
                    eye=dict(x=-1.5, y=-1.5, z=0.5) # Rotate view
                )
            ),
            width=1000,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        html_file = 'gaussian_3d_interactive.html'
        fig_ply.write_html(html_file)
        print(f"Το διαδραστικό γράφημα αποθηκεύτηκε στο {html_file}")
        
    except ImportError:
        print("Η βιβλιοθήκη plotly δεν βρέθηκε. Εγκαταστήστε την με 'pip install plotly' για διαδραστικά γραφήματα.")

if __name__ == "__main__":
    main()
