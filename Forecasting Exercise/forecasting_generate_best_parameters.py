# Randomly sample a subset of the input data for parameter search
def generate_random_sample(input_df, max_size_for_input, dimension_list=[]):
    frac_for_sample = max_size_for_input / len(input_df)
    if len(dimension_list) == 0:
        # If no dimensions, take a simple random sample
        temp_grid_df = input_df.copy().sample(frac=frac_for_sample)
    else:
        # If dimensions provided, take a stratified sample by group
        temp_grid_df = input_df.copy().groupby(by=dimension_list, group_keys=False).sample(frac=frac_for_sample)
    return temp_grid_df


# Generate time-series-friendly CV splits with a minimum training size
def min_train_size_forward_split(X, n_splits=5, min_train_size=100):
    import numpy as np
    
    n_samples = len(X)
    test_size = (n_samples - min_train_size) // n_splits
    if test_size == 0:
        raise ValueError("Not enough data for the requested number of splits and minimum train size.")

    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n_samples:
            break  # stop if we can't form a complete test split

        train_indices = np.arange(train_end)
        test_indices = np.arange(test_start, test_end)

        yield train_indices, test_indices


# Perform hyperparameter search using either Randomized or GridSearch CV
def call_search_cv(search_cv_run_type, tuning_reg, xgb_param_grid, scoring_method, primary_scoring_method, input_df, metric, feature_list, min_cv_split_size):
    import numpy as np
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV
    )
    
    X = input_df[feature_list]
    y = input_df[metric] 

    # Get time series-aware CV splits
    cv_splits = list(min_train_size_forward_split(X, n_splits=5, min_train_size=min_cv_split_size))

    # Choose between RandomizedSearchCV and GridSearchCV
    if search_cv_run_type == 'random':
        grid_search = RandomizedSearchCV(
            estimator=tuning_reg,
            param_distributions=xgb_param_grid,
            n_iter=100,
            cv=cv_splits,
            n_jobs=-1,
            verbose=0,
            scoring=scoring_method,
            refit=primary_scoring_method
        )
    elif search_cv_run_type == 'full':
        grid_search = GridSearchCV(
            estimator=tuning_reg,
            param_grid=xgb_param_grid,
            cv=cv_splits,
            n_jobs=-1,
            verbose=0,
            scoring=scoring_method,
            refit=primary_scoring_method
        )
    else:
        return

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    return best_params


# Main function to generate best parameters through random + grid search
def generate_best_parameters(
    input_df,
    metric,
    feature_list,
    dimension_list=[],
    scoring_method=['neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error'],
    max_size_for_input=100000,
    min_cv_split_size=365,
):
    
    import xgboost as xgb                        
    
    tuning_reg = xgb.XGBRegressor(random_state=42)

    input_df = input_df.reset_index()

    print("Optimizing for: %s" % scoring_method)

    # Handle scoring method input (string or list)
    if isinstance(scoring_method, list):
        primary_scoring_method = scoring_method[0]
        print("%s will be used as primary" % primary_scoring_method)
    elif isinstance(scoring_method, str):
        primary_scoring_method = scoring_method
    else:
        print("ERROR: Invalid scoring method entered")
        return

    input_df_size = len(input_df)

    # Define full parameter grid
    xgb_param_grid = {
        'subsample': [0.8, 0.9, 1.0],
        'n_estimators': [100, 200, 300, 500, 1000, 1500],
        'min_child_weight': [1, 3, 5],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'gamma': [0, 0.1, 0.2, 0.3],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    # Estimate total combinations
    n_iterations = 1
    for values in xgb_param_grid.values():
        n_iterations *= len(values)
    print("Param Grid contains %d combinations of parameters" % n_iterations)

    # Reduced param grid to be populated via randomized search
    xgb_reduced_param_grid = {key: [] for key in xgb_param_grid.keys()}

    num_loops_for_randomized_search = 5
    print("\nStep 1: Fill up a reduced param Grid using Randomized Search")

    for i in range(num_loops_for_randomized_search):
        temp_grid_df = generate_random_sample(input_df, max_size_for_input, dimension_list) \
            if input_df_size >= max_size_for_input else input_df.copy()

        best_params = call_search_cv(
            search_cv_run_type='random',
            tuning_reg=tuning_reg,
            xgb_param_grid=xgb_param_grid,
            scoring_method=scoring_method,
            primary_scoring_method=primary_scoring_method,
            input_df=temp_grid_df,
            metric = metric,
            feature_list=feature_list,
            min_cv_split_size=min_cv_split_size
        )

        # Append selected parameters to reduced grid
        for key in xgb_reduced_param_grid:
            xgb_reduced_param_grid[key].append(best_params[key])

    # De-duplicate values
    for key in xgb_reduced_param_grid:
        xgb_reduced_param_grid[key] = list(set(xgb_reduced_param_grid[key]))

    # Recalculate total combinations in reduced grid
    n_iterations = 1
    for values in xgb_reduced_param_grid.values():
        n_iterations *= len(values)
    print("Reduced Param Grid now contains %d combinations of parameters" % n_iterations)

    # Optional step: Shrink large reduced grid further
    while n_iterations > 200:
        print("\nOptional Step: Reduce size of Param Grid using Randomized Search")

        temp_xgb_reduced_param_grid = {key: [] for key in xgb_param_grid.keys()}

        for i in range(num_loops_for_randomized_search):
            temp_grid_df = generate_random_sample(input_df, max_size_for_input, dimension_list) \
                if input_df_size >= max_size_for_input else input_df.copy()

            best_params = call_search_cv(
                search_cv_run_type='random',
                tuning_reg=tuning_reg,
                xgb_param_grid=xgb_reduced_param_grid,
                scoring_method=scoring_method,
                primary_scoring_method=primary_scoring_method,
                input_df=temp_grid_df,
                metric = metric,
                feature_list=feature_list,
                min_cv_split_size=min_cv_split_size
            )

            for key in temp_xgb_reduced_param_grid:
                temp_xgb_reduced_param_grid[key].append(best_params[key])

        for key in temp_xgb_reduced_param_grid:
            temp_xgb_reduced_param_grid[key] = list(set(temp_xgb_reduced_param_grid[key]))

        xgb_reduced_param_grid = temp_xgb_reduced_param_grid

        # Recompute iteration size
        n_iterations = 1
        for values in xgb_reduced_param_grid.values():
            n_iterations *= len(values)
        print("Reduced Param Grid now contains %d combinations of parameters" % n_iterations)

    # Final step: Full grid search on reduced parameter space
    print("\nStep 2: Run Grid Search on all remaining combinations using Reduced Param Grid")
    temp_grid_df = generate_random_sample(input_df, max_size_for_input, dimension_list) \
        if input_df_size >= max_size_for_input else input_df.copy()

    best_params = call_search_cv(
        search_cv_run_type='full',
        tuning_reg=tuning_reg,
        xgb_param_grid=xgb_reduced_param_grid,
        scoring_method=scoring_method,
        primary_scoring_method=primary_scoring_method,
        input_df=temp_grid_df,
        metric = metric,
        feature_list=feature_list,
        min_cv_split_size=min_cv_split_size
    )

    print(f"Best parameters: {best_params}")
    return best_params
