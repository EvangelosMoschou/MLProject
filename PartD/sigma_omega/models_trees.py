import torch


def get_xgb_dart(n_c, iterations=500):
    from xgboost import XGBClassifier

    use_gpu = torch.cuda.is_available()
    params = dict(
        booster='dart',
        rate_drop=0.1,
        skip_drop=0.5,
        n_estimators=iterations,
        max_depth=6,
        learning_rate=0.05,
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

    return CatBoostClassifier(
        langevin=True,
        diffusion_temperature=1000,
        iterations=iterations,
        depth=8,
        learning_rate=0.03,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        task_type='GPU' if torch.cuda.is_available() else 'CPU',
        verbose=0,
        allow_writing_files=False,
    )
