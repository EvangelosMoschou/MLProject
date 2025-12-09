"""
Μέρος Γ: Ταξινομητής k Κοντινότερων Γειτόνων (KNN)
Αναγνώριση Προτύπων & Μηχανική Μάθηση - 2025-2026
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Φορτώνει δεδομένα από αρχείο CSV.
    Επιστρέφει πίνακα με χαρακτηριστικά και ετικέτες.
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση: {e}")
        return None

def eucl(x, trainData):
    """
    Υπολογίζει την Ευκλείδεια απόσταση από το σημείο x σε όλα τα σημεία του trainData.
    
    Τύπος: d = sqrt((x1-t1)^2 + (x2-t2)^2)
    
    Παράμετροι:
        x: σημείο αναφοράς (2,)
        trainData: δεδομένα εκπαίδευσης (N, 2)
    
    Επιστρέφει:
        Πίνακα αποστάσεων (N,)
    """
    diff = trainData - x
    sq_diff = diff ** 2
    sum_sq = np.sum(sq_diff, axis=1)
    dist = np.sqrt(sum_sq)
    return dist

def neighbors(x, trainData, k):
    """
    Βρίσκει τους k κοντινότερους γείτονες του x στο trainData.
    
    Παράμετροι:
        x: σημείο προς ταξινόμηση (2,)
        trainData: δεδομένα εκπαίδευσης με ετικέτες (N, 3)
        k: αριθμός γειτόνων
    
    Επιστρέφει:
        Τους k κοντινότερους γείτονες (k, 3)
    """
    features = trainData[:, :2]
    distances = eucl(x, features)
    
    # Ταξινόμηση δεικτών κατά αύξουσα απόσταση
    sorted_indices = np.argsort(distances)
    k_indices = sorted_indices[:k]
    
    return trainData[k_indices]

def predict(testData, trainData, k):
    """
    Προβλέπει τις πιθανότητες κλάσης για κάθε σημείο του testData.
    
    Για κάθε σημείο:
    - Βρίσκει τους k γείτονες
    - Μετράει πόσοι ανήκουν σε κάθε κλάση
    - Υπολογίζει πιθανότητες (αθροίζουν στο 1)
    
    Παράμετροι:
        testData: χαρακτηριστικά προς πρόβλεψη (M, 2)
        trainData: δεδομένα εκπαίδευσης (N, 3)
        k: αριθμός γειτόνων
    
    Επιστρέφει:
        Πίνακα πιθανοτήτων (M, 2) [P(κλάση 0), P(κλάση 1)]
    """
    M = testData.shape[0]
    probabilities = np.zeros((M, 2))
    
    for i in range(M):
        x = testData[i]
        k_neighbors = neighbors(x, trainData, k)
        labels = k_neighbors[:, 2]
        
        # Μέτρηση πλειοψηφίας
        count_0 = np.sum(labels == 0)
        count_1 = np.sum(labels == 1)
        
        # Υπολογισμός πιθανοτήτων
        prob_0 = count_0 / k
        prob_1 = count_1 / k
        
        probabilities[i] = [prob_0, prob_1]
        
    return probabilities

def main():
    print("--- Μέρος Γ: Ταξινομητής k Κοντινότερων Γειτόνων ---")
    
    # 1. Φόρτωση δεδομένων
    train_file = '../Datasets/dataset3.csv'
    test_file = '../Datasets/testset.csv'
    
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    if train_data is None or test_data is None:
        return
        
    print(f"Δεδομένα εκπαίδευσης: {train_data.shape}")
    print(f"Δεδομένα ελέγχου: {test_data.shape}")
    
    test_features = test_data[:, :2]
    test_labels = test_data[:, 2]
    
    # 2. Κανονικοποίηση δεδομένων (Z-score)
    # Υπολογισμός μέσου και τ.α. ΜΟΝΟ από τα δεδομένα εκπαίδευσης
    mean = np.mean(train_data[:, :2], axis=0)
    std = np.std(train_data[:, :2], axis=0)
    
    # Εφαρμογή σε εκπαίδευση και έλεγχο
    train_features_norm = (train_data[:, :2] - mean) / std
    test_features_norm = (test_features - mean) / std
    
    # Ανακατασκευή του train_data με κανονικοποιημένα χαρακτηριστικά
    train_data_norm = np.column_stack([train_features_norm, train_data[:, 2]])
    
    print(f"\nΚανονικοποίηση. Μέσος: {mean}, Τ.Α.: {std}")
    
    # 3. Εύρεση βέλτιστου k
    print("\nΕύρεση βέλτιστου k στο [1, 30]...")
    k_values = range(1, 31)
    accuracies = []
    
    for k in k_values:
        # Πρόβλεψη με κανονικοποιημένα δεδομένα
        probs = predict(test_features_norm, train_data_norm, k)
        
        # Επιλογή κλάσης με μέγιστη πιθανότητα
        pred_labels = np.argmax(probs, axis=1)
        
        # Υπολογισμός ακρίβειας
        correct = np.sum(pred_labels == test_labels)
        acc = correct / len(test_labels)
        accuracies.append(acc)
        
    # Εύρεση βέλτιστου k
    best_acc = max(accuracies)
    best_ks = [k for k, acc in zip(k_values, accuracies) if acc == best_acc]
    best_k = best_ks[0]  # Επιλογή του μικρότερου k σε περίπτωση ισοπαλίας
    
    print(f"Βέλτιστο k: {best_k}")
    print(f"Μέγιστη ακρίβεια: {best_acc:.2f}")
    
    # 4. Γράφημα Ακρίβειας vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'b-o')
    plt.plot(best_k, best_acc, 'r*', markersize=15, label=f'Βέλτιστο k={best_k}')
    plt.title('Ακρίβεια KNN vs k (Κανονικοποιημένα Χαρακτηριστικά)')
    plt.xlabel('k')
    plt.ylabel('Ακρίβεια')
    plt.grid(True)
    plt.legend()
    plt.savefig('knn_accuracy.png')
    print("Αποθηκεύτηκε το knn_accuracy.png")
    
    # 5. Όρια Απόφασης (Decision Boundaries)
    print("\nΔημιουργία γραφήματος ορίων απόφασης...")
    
    # Δημιουργία πλέγματος με κανονικοποιημένα χαρακτηριστικά
    features = train_data_norm[:, :2]
    x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
    
    h = 0.05  # Βήμα πλέγματος
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Πρόβλεψη για όλα τα σημεία του πλέγματος
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = predict(grid_points, train_data_norm, best_k)
    Z = np.argmax(probs, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Σχεδίαση
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    
    # Σχεδίαση σημείων εκπαίδευσης
    class_0 = train_data_norm[train_data_norm[:, 2] == 0]
    class_1 = train_data_norm[train_data_norm[:, 2] == 1]
    
    plt.scatter(class_0[:, 0], class_0[:, 1], c='blue', label='Κλάση 0', edgecolors='k')
    plt.scatter(class_1[:, 0], class_1[:, 1], c='red', label='Κλάση 1', edgecolors='k')
    
    plt.title(f'Όρια Απόφασης KNN (k={best_k})')
    plt.xlabel('Χαρακτηριστικό 1 (Κανονικοποιημένο)')
    plt.ylabel('Χαρακτηριστικό 2 (Κανονικοποιημένο)')
    plt.legend()
    plt.savefig('knn_decision_boundary.png')
    print("Αποθηκεύτηκε το knn_decision_boundary.png")

if __name__ == "__main__":
    main()
