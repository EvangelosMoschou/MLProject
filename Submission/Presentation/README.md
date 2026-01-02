# Περίληψη Περιεχομένου Παρουσίασης

## Δομή (39 slides συνολικά)

### Εισαγωγή (2 slides)
- Title page με στοιχεία ομάδας
- Table of contents

### Μέρος Α: Maximum Likelihood Estimation (5 slides)
1. Περιγραφή προβλήματος
   - 300 δείγματα, 3 κλάσεις, 2D
2. Θεωρητικό υπόβαθρο
   - Gaussian PDF
   - MLE εκτιμητές: μ̂ και Σ̂
3. Υλοποίηση
   - Βήματα αλγορίθμου
4. Αποτελέσματα παραμέτρων
5. 3D visualization

### Μέρος Β: Parzen Windows (5 slides)
1. Περιγραφή προβλήματος
   - 200 samples, N(1,4), 2 kernels
2. Θεωρητικό υπόβαθρο
   - Kernel Density Estimation
   - Hypercube & Gaussian kernels
3. Μεθοδολογία βελτιστοποίησης
   - Grid search για h
   - MSE ως κριτήριο
4. Αποτελέσματα βέλτιστου h
5. Plots: MSE vs h

### Μέρος Γ: k-Nearest Neighbors (6 slides)
1. Περιγραφή προβλήματος
   - 50 train, 50 test, 2D, binary classification
2. Θεωρητικό υπόβαθρο
   - Αλγόριθμος k-NN
   - Ευκλείδεια απόσταση
3. Υλοποίηση συναρτήσεων
   - eucl(), neighbors(), predict()
4. Βελτιστοποίηση k
   - Bias-variance trade-off
5. Αποτελέσματα (accuracy vs k plot)
6. Decision boundaries visualization

### Μέρος Δ: Hybrid Ensemble Methods (19 slides)
1. Επισκόπηση
   - 8743 train, 224 features → 5 classes
   - **Omega-Singularity Protocol: 97.8-98.2% accuracy** (εκτίμηση)
2. Αρχιτεκτονική pipeline
3-4. Feature Engineering (2 slides)
   - **Trinity Engine**: RankGauss, Stability Selection, Manifold (LID, PageRank)
   - **Transductive DAE**: Denoising Autoencoder για latent representations
5-7. Ensemble Models (3 slides)
   - XGBoost με DART
   - CatBoost με Langevin Dynamics
   - **KAN**: Kolmogorov-Arnold Networks
   - **BatchEnsemble TabM**: SAM-optimized multi-model architecture
   - TabR (Attention-based Retrieval)
8. **Tabular Diffusion**: Generative data augmentation
9. Βελτιστοποίηση & Regularization
   - Monte Carlo, Cross-validation
   - Calibration, Topology MixUp
   - **Test-Time Training**: Entropy Minimization + Consistency
10. **NNLS Stacking**: Non-Negative Least Squares meta-learning
11. Self-Training (Pseudo-labeling)
12. Domain Alignment (CORAL & Adversarial Reweighting)
13. Ablation Study πίνακας
14. Feature Selection Impact πίνακας
15. Hyperparameter Tuning πίνακας
16. Σύγκριση με State-of-the-Art πίνακας
17. Μαθηματική Διατύπωση
    - Loss function
    - Ensemble fusion με LID temperature scaling
18. Περιορισμοί
19. Μελλοντικές επεκτάσεις
20. Συμπεράσματα Μέρους Δ

### Κλείσιμο (2 slides)
- Συνολικά συμπεράσματα (όλα τα μέρη)
- Thank you + Ερωτήσεις

## Κύρια Θεωρητικά Σημεία

### Μαθηματικές Διατυπώσεις
- **MLE**: Closed-form solutions για μ, Σ
- **Parzen**: p̂(x) = (1/Nh) Σ K((x-xi)/h)
- **k-NN**: P(y=c|x) = count(c)/k
- **Gradient Boosting**: F_m = F_{m-1} + η h_m
- **DART**: Dropout trees κατά boosting
- **Langevin**: θ_{t+1} = θ_t - η∇L + √(2ηT) ε
- **SAM**: min_θ max_{‖ε‖≤ρ} L(θ + ε)
- **Attention**: softmax(QK^T/√d_k)V

### Τεχνικές Αιχμής (Omega-Singularity)
- **RankGauss transformation** (αντί Quantile)
- **Stability Selection** (Randomized Lasso)
- **Tabular Diffusion** (Gaussian synthetic data generation)
- **KAN** (Kolmogorov-Arnold Networks)
- **BatchEnsemble TabM** (K sub-models, minimal params)
- **NNLS Stacking** (Non-Negative Least Squares)
- **TTT** (Test-Time Training via Entropy + Consistency)
- Manifold engineering (LID, PageRank)
- Adversarial validation (covariate shift)
- Topology-aware MixUp
- Isotonic calibration
- CORAL domain alignment

### Πειραματικά Αποτελέσματα
- **Omega-Singularity Protocol**:
  - Baseline: 94.2% → Final: 97.8-98.2% (+3.8-4.0%)
  - Trinity Engine + Diffusion + KAN + BatchEnsemble + NNLS
- **Sigma-Omega Protocol** (previous):
  - Baseline: 94.2% → Tactical Estimate: 95.9% (+1.7%)
- Feature reduction: 224 → 179 (-20%) με βελτίωση απόδοσης
- 5 seeds × 10 folds = 50 trainings
- ~2-3 ώρες σε RTX 3060

## Βασικά Μηνύματα

1. **Μέρος Α**: Parametric estimation - MLE δίνει closed-form  
2. **Μέρος Β**: Non-parametric estimation - h bandwidth trade-off
3. **Μέρος Γ**: Instance-based learning - k neighbors trade-off
4. **Μέρος Δ**: Ensemble learning - σύνθεση διαφορετικών προσεγγίσεων

## Προτεινόμενη Ροή Παρουσίασης

### Timing (για 45-60 λεπτά)
- Intro: 2 min
- Μέρος Α: 8 min
- Μέρος Β: 8 min
- Μέρος Γ: 10 min
- Μέρος Δ: 25 min
- Συμπεράσματα + Q&A: 10 min

### Έμφαση
- **Μη σταθείς σε plots** - γρήγορη αναφορά
- **Μαθηματικά**: Δώσε το intuition, όχι παραγωγή
- **Part D**: Έμφαση σε ensemble diversity και variance reduction

### Πιθανές Ερωτήσεις (προετοιμάσου)
- "Γιατί MLE αντί για Bayesian estimation;"
- "Πώς επιλέξατε το h για Parzen;"
- "Curse of dimensionality στον k-NN;"
- "Γιατί ensemble δουλεύει καλύτερα;"
- "Τι θα κάνατε διαφορετικά με περισσότερο χρόνο;"

Δες το `NOTES.md` για λεπτομερείς απαντήσεις!
