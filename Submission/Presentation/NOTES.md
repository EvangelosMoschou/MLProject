# Σημειώσεις Παρουσίασης - Εργασία ML

## Γενική Επισκόπηση

Η εργασία αποτελείται από 4 μέρη που καλύπτουν βασικές έννοιες της Αναγνώρισης Προτύπων και Μηχανικής Μάθησης:
- **Μέρος Α**: Εκτίμηση παραμέτρων με Maximum Likelihood
- **Μέρος Β**: Non-parametric εκτίμηση με Parzen Windows  
- **Μέρος Γ**: Instance-based learning με k-NN
- **Μέρος Δ**: Advanced ensemble methods για tabular data

---

## Μέρος Α: Maximum Likelihood Estimation (MLE)

### Θεωρητικό Υπόβαθρο

**Στόχος**: Εκτιμούμε τις παραμέτρους πολυδιάστατων κανονικών κατανομών από δεδομένα.

**Κανονική Κατανομή** (Gaussian/Normal):
```
p(x|μ, Σ) = (1 / ((2π)^(d/2) |Σ|^(1/2))) * exp(-1/2 (x-μ)^T Σ^(-1) (x-μ))
```

Όπου:
- **μ**: Μέσος όρος (mean vector) - κέντρο της κατανομής
- **Σ**: Πίνακας συνδιακύμανσης (covariance matrix) - σχήμα/προσανατολισμός
- **d**: Διαστασιμότητα

**Πιθανοφάνεια (Likelihood)**:
```
L(θ|D) = P(D|θ) = ∏ p(x_i|θ)
```

**Log-Likelihood** (για υπολογιστική ευκολία):
```
log L(θ|D) = Σ log p(x_i|θ)
```

**MLE Εκτιμητές** (παράγωγοι των closed-form λύσεων):
```
μ̂ = (1/N) Σ x_i              # Εμπειρικός μέσος
Σ̂ = (1/N) Σ (x_i - μ̂)(x_i - μ̂)^T  # Εμπειρική συνδιακύμανση
```

### Πιθανές Ερωτήσεις & Απαντήσεις

**Ε1: Γιατί χρησιμοποιούμε log-likelihood αντί για likelihood;**
- Α: Γιατί το γινόμενο μετατρέπεται σε άθροισμα → αριθμητική σταθερότητα
- Αποφυγή υποχείλισης (underflow) για μικρές πιθανότητες
- Ευκολότερη παραγώγιση

**Ε2: Είναι οι MLE εκτιμητές unbiased;**
- Α: Ο μέσος όρος μ̂ είναι **unbiased**: E[μ̂] = μ
- Η συνδιακύμανση Σ̂ είναι **biased** (διαιρούμε με N αντί N-1)
- Για unbiased συνδιακύμανση: Σ̂_unbiased = (N/(N-1)) Σ̂

**Ε3: Τι προϋποθέσεις κάνουμε;**
- Α: Τα δεδομένα είναι i.i.d. (ανεξάρτητα και ταυτόσημα κατανεμημένα)
- Κάθε κλάση ακολουθεί κανονική κατανομή
- Yπαρκτός αντίστροφος πίνακας Σ (non-singular)

---

## Μέρος Β: Parzen Windows

### Θεωρητικό Υπόβαθρο

**Πρόβλημα**: Εκτίμηση πυκνότητας πιθανότητας (PDF) χωρίς υπόθεση για τη μορφή της κατανομής.

**Parzen Window (Kernel Density Estimation)**:
```
p̂(x) = (1/(Nh)) Σ K((x - x_i)/h)
```

Όπου:
- **K**: Kernel function (υπερκύβος ή Gaussian)
- **h**: Bandwidth (παράμετρος εξομάλυνσης)
- **N**: Αριθμός δειγμάτων

**Hypercube Kernel**:
```
K(u) = 1  if |u| ≤ 1/2
       0  otherwise
```

**Gaussian Kernel**:
```
K(u) = (1/√(2π)) exp(-u²/2)
```

### Bias-Variance Trade-off

**Μικρό h**:
- Χαμηλό bias (πιστή στα δεδομένα)
- Υψηλή διακύμανση (θόρυβος)
- Over-fitting

**Μεγάλο h**:
- Υψηλό bias (υπερβολική εξομάλυνση)
- Χαμηλή διακύμανση
- Under-fitting

**Βέλτιστο h**: Ισορροπία μεταξύ bias και variance

### Πιθανές Ερωτήσεις & Απαντήσεις

**Ε1: Γιατί το Gaussian kernel είναι καλύτερο από το Hypercube;**
- Α: Το Gaussian είναι **ομαλό** (differentiable παντού)
- Δίνει βάρη που μειώνονται σταδιακά με την απόσταση
- Το Hypercube έχει αιχμηρά όρια (discontinuous)

**Ε2: Πώς επιλέγουμε το h;**
- Α: Cross-validation
- Silverman's rule of thumb: h ≈ 1.06 σ N^(-1/5)
- Grid search (όπως στην εργασία)

**Ε3: Τι συμβαίνει όταν N → ∞;**
- Α: Για σωστό h (h → 0 αλλά Nh → ∞), p̂(x) → p(x) (συνέπεια)

---

## Μέρος Γ: k-Nearest Neighbors (k-NN)

### Θεωρητικό Υπόβαθρο

**Αρχή**: "Τα όμοια παραδείγματα ανήκουν στην ίδια κλάση"

**Αλγόριθμος**:
1. Δεδομένου test point x, βρες τους k πλησιέστερους γείτονες από το training set
2. Υπολόγισε P(y=c|x) = (αριθμός γειτόνων κλάσης c) / k
3. Πρόβλεψη: ŷ = argmax_c P(y=c|x)

**Μετρική Απόστασης** (Ευκλείδεια):
```
d(x_i, x_j) = √(Σ (x_i,d - x_j,d)²)
```

### Επιλογή k

**Μικρό k (π.χ. k=1)**:
- Ευαίσθητο σε θόρυβο
- Περίπλοκα decision boundaries
- Χαμηλό bias, υψηλή διακύμανση

**Μεγάλο k**:
- Εξομάλυνση
- Απλά decision boundaries  
- Υψηλό bias, χαμηλή διακύμανση

**Κανόνας**: k = √N ή cross-validation

### Πιθανές Ερωτήσεις & Απαντήσεις

**Ε1: Είναι ο k-NN parametric ή non-parametric;**
- Α: **Non-parametric** - δεν υποθέτει κατανομή
- Lazy learner - δεν εκπαιδεύει μοντέλο, αποθηκεύει τα δεδομένα

**Ε2: Ποια η υπολογιστική πολυπλοκότητα;**
- Α: Training: O(1) (απλή αποθήκευση)
- Prediction: O(Nd) για κάθε test point (N δείγματα, d διαστάσεις)
- Μπορεί να βελτιωθεί με KD-trees: O(log N)

**Ε3: Τι είναι το curse of dimensionality;**
- Α: Σε υψηλές διαστάσεις, όλα τα σημεία είναι "μακριά"
- Η έννοια της "εγγύτητας" χάνει νόημα
- Χρειάζεται εκθετικά περισσότερα δεδομένα

**Ε4: Γιατί normalization/scaling είναι σημαντικό;**
- Α: Features με μεγάλο range κυριαρχούν στην απόσταση
- Πρέπει να κανονικοποιήσουμε (π.χ. StandardScaler, MinMaxScaler)

---

## Μέρος Δ: Προηγμένες Μέθοδοι Ensemble

### Θεωρητικό Υπόβαθρο

**Ensemble Learning**: Συνδυασμός πολλών μοντέλων για καλύτερη απόδοση

**Θεωρία**:
```
Αν Var(model) = σ², τότε:
Var(average of M independent models) = σ²/M
```

### 1. Gradient Boosting (XGBoost, CatBoost)

**Αρχή**: Aδιτική δόμηση μοντέλων που διορθώνουν τα λάθη των προηγούμενων

**Boosting**:
```
F_m(x) = F_{m-1}(x) + η h_m(x)
```

Όπου:
- **h_m**: Νέο δέντρο εκπαιδευμένο στα υπολείμματα (residuals)
- **η**: Learning rate (shrinkage)

**XGBoost με DART** (Dropouts meet Additive Regression Trees):
- Κατά το training, κάνει **dropout** τυχαίων δέντρων
- Αποφεύγει over-specialization των πρώτων δέντρων
- Παράμετροι: `rate_drop`, `skip_drop`

**CatBoost με Langevin Dynamics**:
- Προσθέτει **Gaussian θόρυβο** στα gradients:
  ```
  θ_{t+1} = θ_t - η∇L + √(2ηT) ε_t
  ```
- **T**: Θερμοκρασία (diffusion temperature)
- Βοηθά στη διαφυγή από τοπικά ελάχιστα

### 2. Neural Networks για Tabular Data

**TabR (Attention-Based Retrieval)**:
1. **Encoder**: Μετατροπή δειγμάτων σε embeddings
2. **Retrieval**: Εύρεση k-NN στον embedding space
3. **Cross-Attention**: Το query δείγμα "προσέχει" τους γείτονές του
4. **Classification**: Final prediction

**Attention Mechanism**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Topology-Aware MixUp**:
- Κλασικό MixUp: x̃ = λx_i + (1-λ)x_j (τυχαία j)
- Topology-Aware: j ∈ k-NN(i) - διατήρηση manifold structure

### 3. Sharpness-Aware Minimization (SAM)

**Πρόβλημα**: Οξέα ελάχιστα (sharp minima) → κακή γενίκευση

**SAM**:
```
min_θ max_{‖ε‖≤ρ} L(θ + ε)
```

Βρίσκει **επίπεδα ελάχιστα** (flat minima) που γενικεύουν καλύτερα

**Αλγόριθμος**:
1. Υπολογισμός gradient: g = ∇L(θ)
2. Perturbation: ε = ρ g/‖g‖
3. Gradient στο perturbed point: g̃ = ∇L(θ + ε)
4. Update: θ ← θ - η g̃

### 4. Feature Engineering

**Quantile Transformation**:
- Μετατροπή σε κανονική κατανομή
- Robust σε outliers
- Βελτιωμένη σύγκλιση

**Manifold Learning**:
- **LID** (Local Intrinsic Dimensionality): Εκτιμά την τοπική διάσταση
- **PageRank**: Εντοπίζει "hub" δείγματα στον KNN γράφο

**Adversarial Validation**:
- Εκπαιδεύει classifier: train vs test
- Reweighting: δίνει υψηλότερο βάρος σε train samples που μοιάζουν με test
- Αντιμετωπίζει **covariate shift**

### 5. Calibration & Variance Reduction

**Isotonic Calibration**:
- Μετατρέπει scores σε καλά βαθμονομημένες πιθανότητες
- **Μονότονη παλινδρόμηση** που ελαχιστοποιεί MSE

**Monte Carlo Ensemble**:
- Πολλαπλά seeds (5 στην υλοποίησή μας)
- Μείωση διακύμανσης: Var/√M

**Cross-Validation**:
- Stratified 10-fold
- Out-of-fold predictions για meta-learning

### 6. Stacking (Meta-Learning)

**Αρχή**: Εκπαίδευση meta-model πάνω σε predictions base models

**Διαδικασία**:
1. Base models κάνουν out-of-fold (OOF) predictions
2. Meta-learner εκπαιδεύεται στα OOF predictions
3. Test predictions: base models → meta-learner

**Πλεονεκτήματα**:
- Μαθαίνει βέλτιστα βάρη
- Καλύτερο από απλό averaging
- Εκμεταλλεύεται συμπληρωματικότητα

### Πιθανές Ερωτήσεις & Απαντήσεις

**Ε1: Γιατί ensemble methods δουλεύουν;**
- Α: **Bias-Variance Decomposition**:
  ```
  Error = Bias² + Variance + Noise
  ```
- Bagging μειώνει variance (π.χ. Random Forest)
- Boosting μειώνει bias (π.χ. XGBoost)
- Ensemble συνδυάζει και τα δύο

**Ε2: Τι είναι το overfitting στο boosting;**
- Α: Πολλά δέντρα → μαθαίνει το θόρυβο
- Αντιμετώπιση:
  - Early stopping (monitor validation loss)
  - Learning rate η < 1
  - Max depth, min samples per leaf
  - DART dropout

**Ε3: Πώς διαλέγουμε μεταξύ XGBoost και CatBoost;**
- Α: **XGBoost**:
  - Γρηγορότερο
  - Καλύτερη τεκμηρίωση
  - Περισσότερες παράμετροι tuning
- **CatBoost**:
  - Automatic categorical feature handling
  - Langevin για καλύτερη γενίκευση
  - Λιγότερο hyperparameter tuning

**Ε4: Τι είναι το covariate shift;**
- Α: Όταν P_train(X) ≠ P_test(X) αλλά P(Y|X) παραμένει ίδια
- Π.χ. διαφορετικά demographics αλλά ίδια σχέση X→Y
- Λύση: Importance reweighting, adversarial validation

**Ε5: Γιατί χρειαζόμαστε calibration;**
- Α: Τα neural networks είναι συχνά **over-confident**
- Calibration διασφαλίζει: predicted probability ≈ true probability
- Σημαντικό για decision making, uncertainty quantification

**Ε6: Τι είναι η διαφορά SAM vs κανονικό SGD;**
- Α: **SGD**: min L(θ) - βρίσκει οποιοδήποτε ελάχιστο
- **SAM**: min max L(θ+ε) - βρίσκει **robust** ελάχιστο
- SAM ελέγχει την "κοιλάδα" γύρω από το ελάχιστο

**Ε7: Πότε να χρησιμοποιήσουμε stacking vs averaging;**
- Α: **Averaging**: Απλό, γρήγορο, δουλεύει καλά αν τα μοντέλα είναι ισοδύναμα
- **Stacking**: Πιο πολύπλοκο, μαθαίνει βάρη, καλύτερο αν τα μοντέλα έχουν διαφορετικές δυνάμεις

---

## Συγκεντρωτική Σύνοψη

### Κλειδιά Επιτυχίας

1. **Feature Engineering**: 
   - Quantile transformation για normalization
   - Feature selection (αφαίρεση θορύβου)
   - Manifold features (LID, PageRank)

2. **Model Diversity**:
   - Tree-based (XGBoost, CatBoost)
   - Neural (TabR, MLP με SAM)
   - Διαφορετικές inductive biases

3. **Variance Reduction**:
   - Monte Carlo (πολλαπλά seeds)
   - Cross-validation
   - Calibration

4. **Advanced Optimization**:
   - DART (dropout for trees)
   - Langevin (exploration via noise)
   - SAM (flat minima)

### Μαθηματικοί Όροι που Πρέπει να Ξέρεις

- **Log-likelihood**
- **Bias-Variance Decomposition**
- **Kernel Density Estimation**
- **Cross-Entropy Loss**
- **Gradient Boosting**
- **Attention Mechanism**
- **Covariate Shift**
- **Isotonic Regression**
- **Stratified Sampling**
- **Out-of-Fold Predictions**

### Τιπς για την Παρουσίαση

1. **Μη μπεις σε πολλές λεπτομέρειες** - δώσε το big picture
2. **Εστίασε στη διαί**σηση** - γιατί επιλέξαμε κάθε τεχνική
3. **Έχε έτοιμα παραδείγματα** - π.χ. "Γιατί DART;" → "Τα πρώτα δέντρα κυριαρχούν"
4. **Μαθηματικά μόνο όπου χρειάζεται** - μην πνιγείς σε τύπους
5. **Αποτελέσματα!** - δείξε ότι δούλεψε (ablation study)

### Πιθανές Δύσκολες Ερωτήσεις

**"Γιατί δεν χρησιμοποιήσατε [μέθοδο X];"**
- Απάντηση: Δοκιμάσαμε X, αλλά Y έδωσε καλύτερα αποτελέσματα γιατί [λόγος]
- Ή: Το X δεν ταίριαζε στο dataset μας γιατί [ιδιαιτερότητα]

**"Πώς αποφύγατε overfitting;"**
- Cross-validation
- Early stopping
- Regularization (dropout, shrinkage)
- Ensemble diversity

**"Πόσο χρόνο πήρε το training;"**
- Να ξέρεις τους χρόνους: ~2-3 ώρες για 5 seeds × 10 folds

**"Τι θα κάνατε διαφορετικά με περισσότερο χρόνο;"**
- Hyperparameter optimization (Optuna)
- Περισσότερα seeds
- Feature engineering automation (featuretools)
- Ensemble με TabPFN, AutoML

---

**Καλή επιτυχία στην παρουσίαση! 🎯**
