# ML Project: Sigma-Omega Protocol (Part D)

Η Εργασία D υλοποιεί το προηγμένο πρωτόκολλο "Sigma-Omega" για ταξινόμηση πινακοποιημένων δεδομένων (tabular classification), συνδυάζοντας σύγχρονες νευρωνικές αρχιτεκτονικές με στοχαστικές μεθόδους βελτιστοποίησης.

## Περιεχόμενα
- [Αρχιτεκτονική](#αρχιτεκτονική-υβριδική-προσέγγιση)
- [Μεθοδολογία Βελτιστοποίησης](#μεθοδολογία-βελτιστοποίησης)
- [Απαιτήσεις](#απαιτήσεις)
- [Εγκατάσταση & Χρήση](#εγκατάσταση--χρήση)
- [Θεωρητικό Υπόβαθρο](#θεωρητικό-υπόβαθρο)
- [Δομή Έργου](#δομή-έργου)
- [Αποτελέσματα](#αποτελέσματα)
- [Αναφορές](#αναφορές)

## Αρχιτεκτονική: Υβριδική Προσέγγιση

Η τελική λύση (`PartD/solution_quantum.py`) ενσωματώνει τρεις κύριες κατηγορίες μοντέλων:

### 1. Νευρωνικά Δίκτυα (Neural Networks)
- **TabR (PyTorch)**: Μοντέλο ανάκτησης βασισμένο σε μηχανισμό προσοχής (attention) που χρησιμοποιεί Cross-Attention για την αξιοποίηση γειτονικών δειγμάτων.
- **ThetaTabM**: Gated MLP εκπαιδευμένο με **SAM (Sharpness-Aware Minimization)** και **Topology MixUp** για βελτιωμένη γενίκευση.

### 2. Δενδρικά Μοντέλα με Στοχαστικές Βελτιώσεις
- **XGBoost DART**: Χρησιμοποιεί dropout στη διαδικασία boosting για αποφυγή υπερπροσαρμογής και δημιουργία πιο ανθεκτικού ensemble.
- **CatBoost Langevin**: Εφαρμόζει Langevin Dynamics για την προσθήκη θερμικού θορύβου στη βελτιστοποίηση, επιτρέποντας την αποφυγή οξέων τοπικών ελαχίστων.

### 3. Μηχανική Χαρακτηριστικών
- **Transductive DAE**: Denoising Autoencoder εκπαιδευμένο στο σύνολο των δεδομένων εκπαίδευσης και δοκιμής για εξομάλυνση του χώρου χαρακτηριστικών.
- **Τοπολογικά Χαρακτηριστικά**: Ενσωμάτωση **LID (Local Intrinsic Dimensionality)** και **PageRank** για την αναπαράσταση της δομής της πολλαπλότητας.

## Μεθοδολογία Βελτιστοποίησης

Η προσέγγιση περιλαμβάνει:
- **Αυτόματη Επιλογή Χαρακτηριστικών**: Αφαίρεση του 20% των χαρακτηριστικών με τη χαμηλότερη συνεισφορά βάσει αρχικής εκτίμησης σημαντικότητας.
- **Monte Carlo Ensemble**: Μέσος όρος προβλέψεων από 5 διαφορετικά seeds (42-46) για μείωση της διακύμανσης κατά ~0.3%.
- **10-Fold Cross-Validation με Isotonic Calibration**: Μεγιστοποίηση των δεδομένων εκπαίδευσης (90% ανά fold) και βαθμονόμηση των πιθανοτήτων για ευθυγράμμιση με πραγματικές συχνότητες σφάλματος.

## Απαιτήσεις

### Υλικό (Hardware)
- **GPU**: NVIDIA GPU με τουλάχιστον 6GB VRAM (βελτιστοποιημένο για RTX 3060)
- **RAM**: Τουλάχιστον 16GB συστήματος
- **Αποθηκευτικός Χώρος**: ~2GB για dependencies και δεδομένα

### Λογισμικό (Software)
```
Python >= 3.10
PyTorch >= 2.0
NumPy >= 1.24
Pandas >= 2.0
scikit-learn >= 1.3
XGBoost >= 2.0
CatBoost >= 1.2
NetworkX >= 3.0
```

Για πλήρη λίστα απαιτήσεων, δείτε το `requirements.txt`.

## Εγκατάσταση & Χρήση

### Εγκατάσταση Dependencies

```bash
# Δημιουργία εικονικού περιβάλλοντος (προαιρετικό αλλά συνιστάται)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ή
venv\Scripts\activate  # Windows

# Εγκατάσταση απαιτήσεων
pip install -r requirements.txt
```

### Εκτέλεση του Πρωτοκόλλου

```bash
# Πλήρη εκτέλεση (5 seeds, 10-fold CV)
python3 PartD/solution_quantum.py

# Για γρήγορο test (1 seed, 3-fold CV) - τροποποιήστε τις μεταβλητές SEEDS και n_splits
```

**Σημείωση**: Το script είναι αυτόνομο (monolithic) και δεν απαιτεί εξωτερικές εξαρτήσεις πέρα από τις βιβλιοθήκες Python.

### Αναμενόμενος Χρόνος Εκτέλεσης
- **GPU (RTX 3060)**: ~2-3 ώρες για πλήρη εκτέλεση
- **CPU**: ~8-12 ώρες

## Θεωρητικό Υπόβαθρο

### Sharpness-Aware Minimization (SAM)
Ο αλγόριθμος SAM βελτιστοποιεί όχι μόνο την τιμή της συνάρτησης απώλειας αλλά και την "οξύτητά" της. Αναζητά περιοχές του χώρου παραμέτρων όπου η απώλεια παραμένει χαμηλή ακόμα και με μικρές διαταραχές, οδηγώντας σε καλύτερη γενίκευση.

### Langevin Dynamics
Προσθέτει στοχαστικό θόρυβο στις ενημερώσεις των gradients, επιτρέποντας στο μοντέλο να "διαφύγει" από στενά τοπικά ελάχιστα. Αποτελεί Bayesian προσέγγιση στη βελτιστοποίηση δενδρικών μοντέλων.

### DART (Dropouts meet Additive Regression Trees)
Κατά την προσθήκη νέου δέντρου, τυχαία δέντρα του ensemble απενεργοποιούνται, αναγκάζοντας το νέο δέντρο να μάθει το πρόβλημα ολιστικά αντί να εστιάζει μόνο στα υπολείμματα.

### Topology MixUp
Αντί για τυχαία ανάμειξη δειγμάτων, η ανάμειξη γίνεται μόνο μεταξύ τοπολογικά γειτονικών σημείων, διατηρώντας τη δομή της πολλαπλότητας (manifold).

## Δομή Έργου

```
MLProject/
├── PartD/
│   ├── src/
│   │   ├── tabr.py              # TabR implementation
│   │   ├── generative.py        # Generative DAE Classifier
│   │   ├── sam.py               # SAM Optimizer
│   │   ├── models.py            # Legacy tree models
│   │   ├── resnet_model.py      # ResNet classifier
│   │   └── data_loader.py       # Data loading utilities
│   ├── legacy/                  # Αρχειοθετημένες εκδόσεις
│   ├── outputs/                 # Αποτελέσματα προβλέψεων
│   ├── solution_quantum.py      # Κύριο script εκτέλεσης
│   └── report.md                # Τεχνική αναφορά (ελληνικά)
├── Datasets/                    # Δεδομένα εκπαίδευσης/δοκιμής
├── Submission/                  # Τελικά αρχεία υποβολής
├── README.md                    # Αυτό το αρχείο
├── walkthrough.md               # Λεπτομερής περιγραφή
└── requirements.txt             # Python dependencies

## Unified CV Engine (Smoke Check)

Για γρήγορο έλεγχο ότι ο Unified CV engine φορτώνει σωστά:

```bash
python3 PartD/sigma_omega/runners/run_cv_engine_smoke.py
```
```

## Αποτελέσματα

Το πρωτόκολλο Sigma-Omega επιτυγχάνει:
- **Μείωση διακύμανσης**: ~0.3% μέσω Monte Carlo ensemble
- **Βελτιωμένη γενίκευση**: SAM και Langevin Dynamics μειώνουν την υπερπροσαρμογή
- **Αποδοτικότητα**: Βελτιστοποίηση για GPU με batch size 2048

### Σύγκριση με Προηγούμενες Εκδόσεις
- **Theta Protocol**: Βασικό ensemble (CatBoost, XGBoost, TabPFN)
- **Omega Protocol**: Προσθήκη DAE και manifold features
- **Epsilon Protocol**: Ενσωμάτωση TabR και SAM
- **Sigma-Omega**: Τελική έκδοση με DART, Langevin, και πλήρη βελτιστοποίηση

## Αναφορές

### Βασικές Δημοσιεύσεις
1. **SAM**: Foret et al. (2021) - "Sharpness-Aware Minimization for Efficiently Improving Generalization"
2. **TabR**: Gorishniy et al. (2022) - "Revisiting Deep Learning Models for Tabular Data"
3. **DART**: Rashmi & Gilad-Bachrach (2015) - "DART: Dropouts meet Multiple Additive Regression Trees"
4. **Langevin Dynamics**: Welling & Teh (2011) - "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
5. **MixUp**: Zhang et al. (2018) - "mixup: Beyond Empirical Risk Minimization"

### Επιπλέον Πόροι
- Λεπτομερής τεχνική ανάλυση: `PartD/report.md`
- Οδηγός υλοποίησης: `walkthrough.md`

---

**Συγγραφέας**: Evangelos Moschou  
**Ημερομηνία**: Ιανουάριος 2026  
**Άδεια**: MIT License