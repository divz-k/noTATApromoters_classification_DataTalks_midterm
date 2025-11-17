#!/usr/bin/env python
# coding: utf-8


def main():
    # Original notebook-style script wrapped inside main()

    from datasets import load_dataset
    ds = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")


    # What is this dataset?
    # 
    # This is a DNA sequence dataset that has been classified into many types: enhancer, promoter, specific histone marks under the label "task". This was curated by InstaDeep, for their Nucleotide transformers paper, from sources like ENCODE etc.
    # The DNA sequence for each task(type) varies in length. For classification into enhancers, the DNA sequence length is 1kb, but for promoters it is 300bp. 
    # 
    # 
    # What do we want to do?
    # 
    # We want to analyse the DNA sequences using the ML models : Linear classification / Decision Trees (Random Forest / XGBoost) and classify them as a "yes/no" into selected categories. I have chosen to work with TATA-less promoters: these are a class of promoters that lack the TATA box recognised by the RNA polymerase, and I would like to see if basic ML models can derive any patterns in the DNA to classifiy them as promoters or not promoters.
    # 
    # 
    # What is required?
    # 
    # The ML models cannot read the DNA sequence: therefore we must convert them into readable data for the model. We need to select parameters/categories from which we can describe this sequence, input them into the model and see if the model can predict 
    # 

    # Unique tasks in the train split
    print(set(ds['train']['task']))

    # Number of examples per task
    from collections import Counter
    print("How many examples per task in the train split?")
    print(Counter(ds['train']['task']))


    # Filter the dataset to get promoter_no_tata data only
    promoters_ds = ds.filter(lambda example: example["task"] == "promoter_no_tata")


    # We need to create categories from the DNA sequence. What can be possible features of the DNA that can be used to classify between promoters and not promoters?
    # 
    # 1) GC content of the promoter
    # 2) Presence of specific 5mers (512 features/kmers, seq and rc counted together)


    import itertools
    import pandas as pd
    import numpy as np


    # motif info database
    complement = str.maketrans('ATGC', 'TACG')
    b = ['A', 'T', 'G', 'C']
    mers5 = [''.join(p) for p in itertools.product(b, repeat=5)]
    rc_mers5 = [''] * int(len(mers5)/2)
    for i, s in enumerate(mers5):
        rc_mers5[i] = s[::-1].translate(complement)
        if rc_mers5[i] in mers5:
            mers5.remove(rc_mers5[i])
    mer5_list = np.column_stack((mers5, rc_mers5))


    def compute_5mer_counts(sequence):
        motif_counts = np.zeros(len(mer5_list), dtype=int)
        for i in range(len(sequence) - 4):
            kmer = sequence[i:i+5]
            idx = np.where(mer5_list == kmer)[0].item()
            motif_counts[idx] += 1
        return motif_counts


    def compute_gc_content(sequence):
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)


    # for a given dataset convert DNA sequence to motif features and GC content, and scale them
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    def makeXY(ds):
        n_samples = len(ds)
        n_features = 512 + 1  # 512 5-mer counts + 1 GC content
        X = np.zeros((n_samples, n_features))  # shape: (samples, features)
        y = []

        for i,currSeq in enumerate(ds):
            seq = currSeq['sequence']
            label = currSeq['label']
            X[i,:] = np.concatenate((compute_5mer_counts(seq), [compute_gc_content(seq)]))
            y.append(label)
        
        X_scaled = scaler.fit_transform(X)
        feature_names = [f'{mer}' for mer in mer5_list[:,0]] + ['gc_content']
        return X_scaled, np.array(y), feature_names


    # Now do Exploratory Data Analysis on the training dataset to see if we have any catchy features differentiating promoters vs non-promoters


    import seaborn as sns
    import matplotlib.pyplot as plt


    # how many promoters vs non-promoters in training data?
    unique, counts = np.unique(y_train, return_counts=True)

    categories = ['Non-Promoter', 'Promoter']
    plt.figure(figsize=(3,2))
    plt.bar(categories, counts, color='skyblue')  # vertical bar plot
    plt.ylabel('#samples')
    plt.title('Class Distribution in Training Data')
    plt.show()


    # no need to split train into train and val as we will use cross-validation


    X_train, y_train, feature_names = makeXY(promoters_ds['train'])
    X_test, y_test, _ = makeXY(promoters_ds['test'])


    # convert X_train, y_train to a df for EDA

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['label'] = y_train

    corrMatrix_all= df_train.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corrMatrix_all, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()


    # given that there are so many samples and features, it is hard to say anything from here. we can just derive that there are some patterns: given that the whole plot doesn't look like noise. this means that some of our input features could have strong correlations (across promoters and non-promoters)


    # what are the correlations within the promoters and non-promoters separately? do we see better clustering for either class?

    promoter_idx = df_train['label'] == 1
    non_promoter_idx = df_train['label'] == 0

    # Promoter clustermap
    g1 = sns.clustermap(
        df_train.loc[promoter_idx, feature_names].corr(),
        cmap='coolwarm',
        vmin=-1, vmax=1,
        linewidths=0.0,
        figsize=(12, 12),
        method='average',
        metric='euclidean'
    )
    g1.fig.suptitle('Promoter Samples Correlation', y=1.02)  # slight bump for spacing

    # Non-Promoter clustermap
    g2 = sns.clustermap(
        df_train.loc[non_promoter_idx, feature_names].corr(),
        vmin=-1, vmax=1,
        cmap='coolwarm',
        linewidths=0.0,
        figsize=(12, 12),
        method='average',
        metric='euclidean'
    )
    g2.fig.suptitle('Non-Promoter Samples Correlation', y=1.02)


    # We can see that the clustering of features is stronger in promoters than non-promoters.
    # 
    # However from this massive plot we can't tell much about the actual features causing the clusters.
    # 
    # What can this mean?
    # 
    # 1) Randomly, there is a correlation among the features that we describe the DNA sequence by. This is depticed by the non-promoter correlation clustering plot.
    # 2) There is a higher than random correlation among the features describing the promoter DNA sequence.
    # 3) This indicates that there are features that are together, more likely to, or less likely to be in the promoter DNA sequence. 
    # 
    # 
    # What can we test next?
    # 1) What features correlate most with each other, by random chance (or biological design, just not based on promoter vs non-promoter)?
    # 2) What are the features that correlate most to being a promoter?
    # 3) what are the least number of features by which, using a regular visualisable plot, we can tell apart promoters and non-promoters?


    # Find the maximum correlation (excluding diagonal)

    corr_no_diag = corrMatrix_all.copy()
    np.fill_diagonal(corr_no_diag.values, np.nan)

    max_corr = corr_no_diag.max().max()
    rows, cols = np.where(corr_no_diag == max_corr)
    for row, col in zip(rows, cols):
        print(f"Features: {corr_no_diag.index[row]} and {corr_no_diag.columns[col]} with correlation {max_corr}")


    # Should we keep these features that have such high correlation? To keep the ML basic and simple, lets not exclude these


    # which features have highest corr with the label?
    corr_with_label = corrMatrix_all['label'].drop('label')
    sortedfeatures = corr_with_label.sort_values(ascending=False)

    plt.figure(figsize=(10,3))
    plt.bar(sortedfeatures.index, sortedfeatures.values)
    plt.title('Feature Correlation with Label- Distribution')
    plt.ylabel('Correlation')
    plt.ylim([-1,1])

    plt.xticks([])
    plt.xlabel('513 Features- Sorted by Correlation')
    plt.show()


    # Roughly half the features are positively and half are negatively correlated to being a promoter. Lets try other metrics, like the mutual information score


    # mutual information of each feature with the label
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X_train, y_train, discrete_features=True, random_state=1)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10,3))
    plt.bar(mi_series.index, mi_series.values, color='orange')
    plt.title('Feature Mutual Information with Label- Distribution')
    plt.ylabel('Mutual Information')
    plt.xticks([])
    plt.xlabel('513 Features- Sorted by Mutual Information')
    plt.show()


    # The MI distribution looks similar to the correlation, as expected.
    # 
    # How do these featurees coorelate with respect to each other? Lets see how many Principal componenets are required to descrive all the features?


    from sklearn.decomposition import PCA

    pca = PCA().fit(df_train[feature_names])
    plt.plot((pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Fraction of Variance")
    plt.title("PCA - Cumulative Variance Explained")
    plt.plot( np.full(100, 10), np.linspace(0, 0.2, 100), linestyle='dashed', color='grey', label='10 components')
    plt.legend()
    plt.show()


    # The dotted line deptics the top 10 PCs: this tells us that all the features can simply be described with 10 PCs: they are highly dependent on each other. But as said earlier, let us keep all the features to be simple for our standard ML model.
    # 
    # Can the top 2 PCs tell apart promoter vs non-promoter?


    # PCA scatter plot of first 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)  

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['label'] = y_train

    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='label', palette='Set1', alpha=0.5)
    plt.title('PCA Scatter Plot')
    plt.show()


    # Yes, it looks like the top 2 PCs can tell apart promoter and non-promoter. But can we say the same about just the most correlated and anticorrelated feature? Or the top 2 MIs?


    # top 2 MIs vs promoter/non-promoter

    df_top2mi = df_train[[mi_series.index[0], mi_series.index[1], 'label']]
    sns.scatterplot(data=df_top2mi, x=mi_series.index[0], y=mi_series.index[1], hue='label', palette='Set1', alpha=0.5)
    plt.title('Top 2 MI Features Scatter Plot')
    plt.show()


    # Maybe well, but not as well as the PCs. This is because GC content and GGCGG themselves are highly correlated. Lets check out top correlated and anticorrelated features.


    max_feat = sortedfeatures.idxmax()  # feature with highest correlation
    min_feat = sortedfeatures.idxmin()  # feature with lowest correlation

    print("Most correlated feature:", max_feat, "with corr:", sortedfeatures[max_feat])
    print("Most anticorrelated feature:", min_feat, "with corr:", sortedfeatures[min_feat])
    # Scatter plot
    plt.scatter(df_train[max_feat], df_train[min_feat], c=df_train['label'], cmap='coolwarm', alpha=0.5)
    plt.xlabel(f"{max_feat} (max corr feature)")
    plt.ylabel(f"{min_feat} (max anti-corr feature)")
    plt.colorbar(label='Correlation value')
    plt.show()




    # Looks similar to the MI: the most correlated and anticorrelated are anticorrelated themselves. As all these features are so highly dependent on each other- we will get only this from the EDA. Lets just take them all and try different models.
    # 
    # What models to try?
    # 1) Linear classifier: try different c: [0.001, 0.01, 0.1, 1, 10, 100]
    # 2) Decision Tree:
    # 3) Random Forest
    # 4) XGBoost
    # 


    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score, accuracy_score

    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'negRMSE' : 'neg_mean_squared_error' #we need to do this because GridSearchCV always tries to maximize the metric, and RMSE needs to be minimized
    }

    model = LogisticRegression(
        penalty='l2',        
        solver='liblinear',  
        max_iter=1000,
        random_state=42
    )

    # tune C:
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                
        scoring=scoring,
        refit='roc_auc', #pick best model based on ROC AUC
    )

    grid.fit(X_train, y_train)

    print("Best C:", grid.best_params_)
    print("Best model (based on roc_auc):", grid.best_score_)

    # access all metrics for each C
    cv_results_regression = pd.DataFrame(grid.cv_results_)


    cv_results_regression.T


    best_model_regression = grid.best_estimator_

    y_pred = best_model_regression.predict(X_test)
    y_pred_proba = best_model_regression.predict_proba(X_test)[:, 1]

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Test F1 Score:", f1_score(y_test, y_pred))
    print("Test RMSE:", mean_squared_error(y_test, y_pred))


    # Ok- so we have the best possible regression model: with parameters:
    # 
    # Best C: {'C': 0.001}
    # 
    # Best model (based on roc_auc): 0.931
    # 
    # 
    # Test Accuracy: 0.855
    # 
    # Test ROC AUC: 0.936
    # 
    # Test F1 Score: 0.849
    # 
    # Test RMSE: 0.144
    # 
    # 
    # Let us try Decision Tree next


    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(random_state=1)

    param_grid_tree = {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    grid_tree = GridSearchCV(
        estimator=tree,
        param_grid=param_grid_tree,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )

    grid_tree.fit(X_train, y_train)
    print("Best C:", grid_tree.best_params_)
    print("Best model (based on roc_auc):", grid_tree.best_score_)

    # access all metrics for each C
    cv_results_tree = pd.DataFrame(grid_tree.cv_results_)


    best_model_tree = grid_tree.best_estimator_

    y_pred = best_model_tree.predict(X_test)
    y_pred_proba = best_model_tree.predict_proba(X_test)[:, 1]

    print("Tree Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Tree Test ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Tree Test F1 Score:", f1_score(y_test, y_pred))
    print("Tree Test RMSE:", mean_squared_error(y_test, y_pred))

    cv_results_tree


    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid_forest = { #selected params based on prior tuning on the tree
        'n_estimators': [10, 50, 100],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [5],
        'max_features': [None]
    }

    grid_forest = GridSearchCV(
        estimator=forest,
        param_grid=param_grid_forest,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )

    grid_forest.fit(X_train, y_train)
    print("Best params:", grid_forest.best_params_)
    print("Best model (based on roc_auc):", grid_forest.best_score_)

    # access all metrics for each parameter combination
    cv_results_forest = pd.DataFrame(grid_forest.cv_results_)


    best_model_forest = grid_forest.best_estimator_

    y_pred = best_model_forest.predict(X_test)
    y_pred_proba = best_model_forest.predict_proba(X_test)[:, 1]

    print("Forest Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Forest Test ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Forest Test F1 Score:", f1_score(y_test, y_pred))
    print("Forest Test RMSE:", mean_squared_error(y_test, y_pred))

    cv_results_forest


    import xgboost as xgb

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    param_grid_xgb = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 1],
        'colsample_bytree': [0.7, 1],
    }

    grid_xgb = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid_xgb,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )

    grid_xgb.fit(X_train, y_train)
    print("Best params:", grid_xgb.best_params_)
    print("Best model (based on roc_auc):", grid_xgb.best_score_)

    best_model_xgb = grid_xgb.best_estimator_

    y_pred = best_model_xgb.predict(X_test)
    y_pred_proba = best_model_xgb.predict_proba(X_test)[:, 1]

    print("XGB Test Accuracy:", accuracy_score(y_test, y_pred))
    print("XGB Test ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("XGB Test F1 Score:", f1_score(y_test, y_pred))
    print("XGB Test RMSE:", mean_squared_error(y_test, y_pred))

    cv_results_xgb = pd.DataFrame(grid_xgb.cv_results_)


    import pandas as pd

    testScores = [
        {
            'Model': 'XGB',
            'Test Accuracy': 0.841,
            'Test ROC AUC': 0.934,
            'Test F1 Score': 0.826,
            'Test RMSE': 0.158
        },
        {
            'Model': 'Regression',
            'Test Accuracy': 0.855,
            'Test ROC AUC': 0.936,
            'Test F1 Score': 0.849,
            'Test RMSE': 0.144
        },
        {
            'Model': 'Forest',
            'Test Accuracy': round(0.8345481049562682, 3),
            'Test ROC AUC': 0.922,
            'Test F1 Score': 0.823,
            'Test RMSE': round(0.16545189504373178, 3)
        },
        {
            'Model': 'Tree',
            'Test Accuracy': round(0.8236151603498543, 3),
            'Test ROC AUC': round(0.9127276899931152, 3),
            'Test F1 Score': 0.810,
            'Test RMSE': round(0.17638483965014579, 3)
        }
    ]

    # Convert to DataFrame
    df_testScores = pd.DataFrame(testScores)

    df_testScores 
    plt.figure(figsize=(2.5,2.5))
    sns.barplot(x='Model', y='Test ROC AUC', data=df_testScores, palette='viridis')
    plt.title('Test ROC AUC by Model')
    plt.ylim(0, 1)
    plt.show()


    # We can see that all 4 tested models perform similarly, but if I have to pick one, I will choose to save the regression one as my main model. 


    import joblib

    # Suppose you have trained models
    xgb_model = grid_xgb.best_estimator_   
    forest_model = grid_forest.best_estimator_
    tree_model = grid_tree.best_estimator_ 
    regression_model = grid.best_estimator_

    # Save models to files
    joblib.dump(xgb_model, "xgb_model.pkl")
    joblib.dump(forest_model, "forest_model.pkl")
    joblib.dump(tree_model, "tree_model.pkl")
    joblib.dump(regression_model, "regression_model.pkl")


if __name__ == "__main__":
    main()
