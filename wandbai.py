#pip install wandb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.8, random_state = 357) #se puede cambiar el tamaño del test para ver como cambia

hyperparameters = {
    'learning_rates': [0.01, 0.1, 0.2, 0.25],
    'max_depths': [2, 3, 4, 5],
    'n_estimators': [50, 60, 70],
    'loss_functions': ['deviance'], #otras son log_loss, squared_error
    'subsamples': [0.8, 1.0], #una tasa mas baja previene el sobreajuste pero influye en la varianza del modelo
    'min_samples_splits': [2, 4], 
    'min_samples_leafs': [1, 2]
}

for lr in hyperparameters['learning_rates']:
    for max_depth in hyperparameters['max_depths']:
        for n_estimator in hyperparameters['n_estimators']:
            for loss_function in hyperparameters['loss_functions']:
                for subsample in hyperparameters['subsamples']:
                    for min_samples_split in hyperparameters['min_samples_splits']:
                        for min_samples_leaf in hyperparameters['min_samples_leafs']:
                          experiment_name = f"gbm_lr{lr}_depth{max_depth}_est{n_estimator}_loss{loss_function}_subsample{subsample}_minsplit{min_samples_split}_minleaf{min_samples_leaf}"
                          wandb.init(project="wine-project", name=experiment_name, config=hyperparameters)


                          clf = GradientBoostingClassifier(learning_rate=lr, max_depth=max_depth, n_estimators=n_estimator,
                                                             loss=loss_function, subsample=subsample, min_samples_split=min_samples_split,
                                                             min_samples_leaf=min_samples_leaf, random_state=357, validation_fraction=0.1,
                                                             n_iter_no_change=5, tol=0.01)
                            
                          clf.fit(X_train, y_train)

                            # Make predictions
                          y_pred = clf.predict(X_test)
                          y_pred_proba = clf.predict_proba(X_test)

                            # Log hyperparameters to wandb
                          wandb.config.learning_rate = lr
                          wandb.config.max_depth = max_depth
                          wandb.config.n_estimators = n_estimator
                          wandb.config.loss_function = loss_function
                          wandb.config.subsample = subsample
                          wandb.config.min_samples_split = min_samples_split
                          wandb.config.min_samples_leaf = min_samples_leaf

                            # Calculate performance metrics
                          accuracy = accuracy_score(y_test, y_pred)
                          f1_macro = f1_score(y_test, y_pred, average="macro")
                          y_test_bin = label_binarize(y_test, classes=np.unique(y))
                          y_pred_bin = y_pred_proba.reshape(-1, len(np.unique(y)))
                          roc_auc_macro = roc_auc_score(y_test_bin, y_pred_bin, average="macro", multi_class="ovr")

                            # Log metrics to wandb
                          wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "roc_auc_macro": roc_auc_macro,
                                       "validation_score": clf.train_score_[-1]})

                            # Finish the experiment
                          wandb.finish()