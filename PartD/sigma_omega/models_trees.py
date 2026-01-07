import torch



def get_xgb_dart(n_c, iterations=500):
    from xgboost import XGBClassifier

    use_gpu = torch.cuda.is_available()
    # Optimized Params (Jan 6 Run)
    # n_estimators overridden by iterations arg
    params = dict(
        booster='dart',
        rate_drop=0.1,
        skip_drop=0.5,
        n_estimators=iterations,
        max_depth=8,  # Optimized from 6
        learning_rate=0.066, # Optimized
        subsample=0.96, # Optimized
        colsample_bytree=0.62, # Optimized
        gamma=0.016, # Optimized
        reg_alpha=0.0009, # Optimized
        reg_lambda=1.14, # Optimized
        objective='multi:softprob',
        num_class=int(n_c),
        eval_metric='mlogloss',
        verbosity=0,
    )
    if use_gpu:
        # XGBoost >= 2.0 uses device='cuda' and tree_method='hist'
        params.update(tree_method='hist', device='cuda')
    else:
        params.update(tree_method='hist', device='cpu')
    return XGBClassifier(**params)


def get_cat_langevin(n_c, iterations=1000):
    from catboost import CatBoostClassifier

    # Optimized Params (Jan 6 Run)
    return CatBoostClassifier(
        langevin=True,
        diffusion_temperature=1000,
        iterations=iterations, # Typically 1000
        depth=9, # Optimized from 8
        learning_rate=0.0485, # Optimized
        l2_leaf_reg=2.68, # Optimized
        random_strength=2.42, # Optimized
        bagging_temperature=0.035, # Optimized
        loss_function='MultiClass',
        eval_metric='MultiClass',
        task_type='GPU' if torch.cuda.is_available() else 'CPU',
        verbose=0,
        allow_writing_files=False,
    )
