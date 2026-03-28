"""ASDclassification.py

This module implements classification models for Autism Spectrum Disorder (ASD)
using machine learning techniques. It includes data preprocessing, model training,
and evaluation functions.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
import os
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class GazeClassificationPipeline:
    """
    Complete pipeline for gaze data classification using KNN and Random Forest evaluation.
    """
    
    def __init__(self, use_gridsearch=True, rf_cv_splits=5, create_visualizations=True, rf_use_loo=True):
        """
        Initialize the pipeline.
        
        Args:
            use_gridsearch (bool): Whether to use grid search for KNN hyperparameter optimization
            rf_cv_splits (int): Number of splits for Random Forest cross-validation (ignored if rf_use_loo=True)
            create_visualizations (bool): Whether to create visualization plots
            rf_use_loo (bool): Whether to use leave-one-out CV for Random Forest (default: True)
        """
        self.use_gridsearch = use_gridsearch
        self.rf_cv_splits = rf_cv_splits
        self.rf_use_loo = rf_use_loo
        self.create_visualizations = create_visualizations
        self.best_params_per_stimulus = {}
        self.knn_results = None
        self.rf_results = None
        self.stimulus_metrics = {}
        
        # Create visualizations directory if needed
        if self.create_visualizations:
            self.viz_dir = Path("visualizations")
            self.viz_dir.mkdir(exist_ok=True)
    
    def find_best_knn_parameters(self, X_train, y_train, groups_train):
        """
        Find the best hyperparameters for KNN classifier using GridSearchCV
        with group-based cross-validation on TRAINING DATA ONLY.
        
        Args:
            X_train (array): Training feature matrix
            y_train (array): Training target labels
            groups_train (array): Training group labels for cross-validation
        
        Returns:
            dict: Best parameters found
            float: Best score achieved
        """
        print("Performing hyperparameter optimization...")
        
        # Define the parameter grid to search
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9, 11], 
            'weights': ['uniform', 'distance'], 
            'metric': ['euclidean', 'manhattan'] 
        }
        
        # Initialize KNN classifier
        knn = KNeighborsClassifier()
        
        # Set up GroupKFold for cross-validation (using only training groups)
        gkf = GroupKFold(n_splits=len(set(groups_train))) 
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=gkf.split(X_train, y_train, groups_train),
            scoring='accuracy',
            #n_jobs=1,  # Use 1 job for grid search to avoid memory issues
            n_jobs=-1, # Uncomment to use all available cores
            verbose=1
        )
        
        # Perform the grid search
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_

    def process_stimulus(self, df_stim, stimulus_name):
        """
        Process a single stimulus by performing leave-one-subject-out cross-validation
        with KNN classification.
        
        Args:
            df_stim (DataFrame): DataFrame filtered for a specific stimulus
            stimulus_name (str): Name of the stimulus being processed
        
        Returns:
            tuple: (DataFrame with results, dict with best parameters if grid search was used)
        """
        print(f"\nProcessing stimulus: {stimulus_name}")
        start_time = time.time()
        
        # Features, labels, and groups
        X = df_stim[["X", "Y"]].values  # Gaze coordinates
        y = df_stim["Class"].values     # Target classes ('C' or 'P')
        groups = df_stim["Child ID"].values  # Child identifiers for group CV
        
        # Group k-fold cross-validation (leave-one-subject-out)
        gkf = GroupKFold(n_splits=len(set(groups)))
        
        child_predictions = []  # Store probability predictions for each child
        child_ids = []          # Store corresponding child IDs
        child_true_labels = []  # Store true labels for evaluation

        # Store parameters for all folds
        fold_best_params = []  # Store parameters for each fold
        foldnb = 0 #MGV
        
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            test_child_ids = groups[test_idx]
            
            foldnb =+ 1 #MGV
            print("Processing " + str(stimulus_name) + " and fold nb " + str(foldnb)) #MGV

            # Verify that test set contains only one child
            unique_test_ids = set(test_child_ids)
            if len(unique_test_ids) != 1:
                raise ValueError(f"Error: test set contains more than one child: {unique_test_ids}")
            
            # Optionally perform grid search to find best parameters (NESTED CV)
            if self.use_gridsearch:
                current_fold_params, _ = self.find_best_knn_parameters(X_train, y_train, groups_train)
                fold_best_params.append({
                    'fold': fold_idx,
                    'child_id': test_child_ids[0],
                    'params': current_fold_params
                })
                knn = KNeighborsClassifier(**current_fold_params)
            else:
                # Use default parameters
                knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
                fold_best_params.append({
                    'fold': fold_idx,
                    'child_id': test_child_ids[0],
                    'params': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'minkowski'}
                })
            
            # Train the model
            knn.fit(X_train, y_train)
            
            # Make predictions
            predictions = knn.predict(X_test)
            
            # Get most frequent true label for this child (for evaluation)
            true_label_counts = Counter(y_test)
            most_common_true_label = true_label_counts.most_common(1)[0][0]
            
            # Calculate probability of class 'P' for this child
            child_id = test_child_ids[0]
            counts = Counter(predictions)
            count_C = counts.get('C', 0)  # Count of 'C' predictions
            count_P = counts.get('P', 0)  # Count of 'P' predictions
            total_points = count_C + count_P
            
            # Calculate probability (avoid division by zero)
            probability_P = count_P / total_points if total_points > 0 else 0.5
            
            # Store results
            child_ids.append(child_id)
            child_predictions.append(probability_P)
            child_true_labels.append(most_common_true_label)
        
        # Report parameter distribution 
        if self.use_gridsearch and fold_best_params:
            # Calculate the most common parameters for this stimulus
            param_counts = {}
            for fold_info in fold_best_params:
                params = fold_info['params']
                param_key = (params['n_neighbors'], params['weights'], params['metric'])
                param_counts[param_key] = param_counts.get(param_key, 0) + 1
            
            # Take the most common combination
            most_common_params = max(param_counts, key=param_counts.get)
            self.best_params_per_stimulus[stimulus_name] = {
                'n_neighbors': most_common_params[0],
                'weights': most_common_params[1],
                'metric': most_common_params[2],
                'frequency': param_counts[most_common_params],
                'total_folds': len(fold_best_params)
            }
            
            print("Parameter distribution across folds:")
            param_summary = {}
            for fold_info in fold_best_params:
                params = fold_info['params']
                for key, value in params.items():
                    if key not in param_summary:
                        param_summary[key] = []
                    param_summary[key].append(value)
            
            for param, values in param_summary.items():
                unique_vals, counts = np.unique(values, return_counts=True)
                print(f"  {param}: {dict(zip(unique_vals, counts))}")
        else:
            print("Using default parameters: n_neighbors=5, weights='distance'")
        
        # Convert probabilities to class predictions for evaluation
        binary_predictions = ['P' if p > 0.5 else 'C' for p in child_predictions]
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(child_true_labels, binary_predictions)
        f1 = f1_score(child_true_labels, binary_predictions, pos_label='P', average='binary')
        
        # Calculate confusion matrix for this stimulus
        cm = confusion_matrix(child_true_labels, binary_predictions, labels=['C', 'P'])
        
        # Store metrics for this stimulus
        self.stimulus_metrics[stimulus_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_labels': child_true_labels,
            'predictions': binary_predictions
        }
        
        print(f"Stimulus '{stimulus_name}' metrics:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Processing time: {time.time() - start_time:.2f} seconds")
        
        # Create visualization for this stimulus
        if self.create_visualizations:
            self.create_stimulus_visualization(stimulus_name, cm, accuracy)
        
        # Create DataFrame with results for this stimulus
        results_df = pd.DataFrame({
            "Child ID": child_ids,
            f"prediction_{stimulus_name}": child_predictions
        })
        
        return results_df, fold_best_params
        
    def create_stimulus_visualization(self, stimulus_name, confusion_matrix, accuracy):
        """
        Create and save confusion matrix visualization for a stimulus.
        
        Args:
            stimulus_name (str): Name of the stimulus
            confusion_matrix (array): Confusion matrix for the stimulus
            accuracy (float): Accuracy score for the stimulus

        Output Files:
            Saves PNG image: {viz_dir}/confusion_matrix_{safe_stimulus_name}.png
            Contains heatmap visualization of confusion matrix with accuracy caption
            Format: 8x6 inch figure at 300 DPI with tight bounding box
        """
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Control (C)', 'Positive (P)'],
                   yticklabels=['Control (C)', 'Positive (P)'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {stimulus_name}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Add accuracy as caption
        plt.figtext(0.5, 0.02, f'Accuracy: {accuracy:.4f}', 
                   ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        safe_filename = "".join(c for c in stimulus_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        plt.savefig(self.viz_dir / f'confusion_matrix_{safe_filename}.png', 
                   dpi=300, bbox_inches='tight')
        time.sleep(0.5)
        plt.close()
    
    def create_accuracy_summary_plot(self):
        """
        Create a summary plot showing accuracy for all stimuli.

        Output Files:
            Saves PNG image: {viz_dir}/accuracy_summary.png
            Contains dual subplot figure (12x10 inches, 300 DPI):
            * Top plot: Bar chart of accuracy by stimulus
            * Bottom plot: Bar chart of F1 scores by stimulus
            Both plots include value labels and grid lines
        """
        if not self.stimulus_metrics:
            print("No stimulus metrics available for visualization.")
            return
        
        # Extract data for plotting
        stimuli = list(self.stimulus_metrics.keys())
        accuracies = [self.stimulus_metrics[stim]['accuracy'] for stim in stimuli]
        f1_scores = [self.stimulus_metrics[stim]['f1_score'] for stim in stimuli]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Accuracy plot
        bars1 = ax1.bar(range(len(stimuli)), accuracies, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Stimuli')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy by Stimulus')
        ax1.set_xticks(range(len(stimuli)))
        ax1.set_xticklabels(stimuli, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1 Score plot
        bars2 = ax2.bar(range(len(stimuli)), f1_scores, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Stimuli')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score by Stimulus')
        ax2.set_xticks(range(len(stimuli)))
        ax2.set_xticklabels(stimuli, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, f1) in enumerate(zip(bars2, f1_scores)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'accuracy_summary.png', dpi=300, bbox_inches='tight')
        time.sleep(0.5)
        plt.close()
        
        print(f"Summary accuracy plot saved to: {self.viz_dir / 'accuracy_summary.png'}")
    
    def create_parameter_distribution_plots(self):
        """
        Create visualizations showing the distribution of best parameters across stimuli.

        Output Files:
            Saves PNG image: {viz_dir}/parameter_distributions.png
            Contains 2x2 subplot figure (15x12 inches, 300 DPI):
            * Top-left: Distribution of optimal k values (bar chart)
            * Top-right: Distribution of optimal weight types (bar chart)  
            * Bottom-left: Distribution of optimal distance metrics (bar chart)
            * Bottom-right: Top parameter combinations with fold frequency info (horizontal bar chart)
        
        Side Effects:
            Calls create_parameter_table() which generates:
            - CSV file: {viz_dir}/parameter_performance_table.csv
            - PNG image: {viz_dir}/parameter_performance_analysis.png

        Note:
            Only creates visualizations if grid search was used (best_params_per_stimulus not empty).
            Parameter combinations show format: "k=X, weight_type, metric (frequency/total_folds folds)"
        """
        if not self.best_params_per_stimulus:
            print("No parameter data available for visualization (grid search not used).")
            return
        
        # Extract parameter data - need to get the most common parameters for each stimulus
        stimuli = list(self.best_params_per_stimulus.keys())
        n_neighbors = []
        weights = []
        metrics = []
        
        # Calculate most common parameters for each stimulus
        for stim in stimuli:
            fold_params = self.best_params_per_stimulus[stim]
            
            # Extract parameters from all folds for this stimulus
            stim_n_neighbors = [fold_info['params']['n_neighbors'] for fold_info in fold_params]
            stim_weights = [fold_info['params']['weights'] for fold_info in fold_params]
            stim_metrics = [fold_info['params']['metric'] for fold_info in fold_params]
            
            # Find most common parameter for each type
            from collections import Counter
            most_common_k = Counter(stim_n_neighbors).most_common(1)[0][0]
            most_common_weight = Counter(stim_weights).most_common(1)[0][0]
            most_common_metric = Counter(stim_metrics).most_common(1)[0][0]
            
            n_neighbors.append(most_common_k)
            weights.append(most_common_weight)
            metrics.append(most_common_metric)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of n_neighbors
        unique_k, counts_k = np.unique(n_neighbors, return_counts=True)
        bars1 = ax1.bar(unique_k, counts_k, color='lightblue', alpha=0.7, edgecolor='navy')
        ax1.set_xlabel('Number of Neighbors (k)')
        ax1.set_ylabel('Frequency (Most Common per Stimulus)')
        ax1.set_title('Distribution of Most Common k Values Across Stimuli')
        ax1.set_xticks(unique_k)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars1, counts_k):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(count), ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 2. Distribution of weights
        unique_weights, counts_weights = np.unique(weights, return_counts=True)
        colors_weights = ['lightcoral', 'lightgreen'][:len(unique_weights)]
        bars2 = ax2.bar(unique_weights, counts_weights, color=colors_weights, alpha=0.7, edgecolor='darkred')
        ax2.set_xlabel('Weight Type')
        ax2.set_ylabel('Frequency (Most Common per Stimulus)')
        ax2.set_title('Distribution of Most Common Weight Types Across Stimuli')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars2, counts_weights):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(count), ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Distribution of metrics
        unique_metrics, counts_metrics = np.unique(metrics, return_counts=True)
        colors_metrics = ['gold', 'orange'][:len(unique_metrics)]
        bars3 = ax3.bar(unique_metrics, counts_metrics, color=colors_metrics, alpha=0.7, edgecolor='darkorange')
        ax3.set_xlabel('Distance Metric')
        ax3.set_ylabel('Frequency (Most Common per Stimulus)')
        ax3.set_title('Distribution of Most Common Distance Metrics Across Stimuli')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars3, counts_metrics):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(count), ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 4. Parameter combinations with frequency information
        param_combinations = {}

        for stim in stimuli:
            stim_params = self.best_params_per_stimulus[stim]
            
            # Create combination string without fold-specific frequency
            combo = f"k={stim_params['n_neighbors']}, {stim_params['weights']}, {stim_params['metric']}"
            
            # Count how many stimuli use this combination
            param_combinations[combo] = param_combinations.get(combo, 0) + 1

        # Sort combinations by frequency (number of stimuli using them)
        sorted_combos = sorted(param_combinations.items(), key=lambda x: x[1], reverse=True)
        combo_names = [combo[0] for combo in sorted_combos[:10]]  # Top 10 combinations
        combo_counts = [combo[1] for combo in sorted_combos[:10]]

        if combo_names:
            bars4 = ax4.barh(range(len(combo_names)), combo_counts, color='mediumpurple', alpha=0.7)
            ax4.set_xlabel('Number of Stimuli')
            ax4.set_ylabel('Parameter Combinations')
            ax4.set_title('Top Parameter Combinations Across Stimuli')
            ax4.set_yticks(range(len(combo_names)))
            ax4.set_yticklabels(combo_names, fontsize=8)
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars4, combo_counts)):
                ax4.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                        str(count), ha='left', va='center', fontsize=9, weight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
        time.sleep(0.5)
        plt.close()
        
        print(f"Parameter distribution plots saved to: {self.viz_dir / 'parameter_distributions.png'}")
        
        # Create detailed parameter table
        self.create_parameter_table()
    
    def create_parameter_table(self):
        """
        Create a detailed table showing parameters and performance for each stimulus.
                
        Output Files:
            Saves CSV file: {viz_dir}/parameter_performance_table.csv
            Contains columns: 
            - Stimulus: stimulus name
            - k_neighbors: optimal number of neighbors
            - weights: optimal weight type ('uniform' or 'distance')
            - metric: optimal distance metric ('euclidean', 'manhattan', etc.)
            - param_frequency: how many folds used these parameters
            - total_folds: total number of folds for this stimulus
            - param_consistency: ratio of param_frequency/total_folds
            - accuracy: classification accuracy for this stimulus
            - f1_score: F1 score for this stimulus
            Sorted by accuracy (descending), float values formatted to 4 decimal places
        
        Side Effects:
            Calls create_parameter_performance_plot() which generates:
            PNG image: {viz_dir}/parameter_performance_analysis.png
            Contains 2x2 subplot figure (15x12 inches, 300 DPI):
            * Top-left: Accuracy vs Number of Neighbors (scatter with trend line)
            * Top-right: F1 Score vs Number of Neighbors (scatter with trend line)  
            * Bottom-left: Accuracy distribution by Weight Type (box plots)
            * Bottom-right: Accuracy distribution by Distance Metric (box plots)
        
        Requirements:
            Requires both best_params_per_stimulus and stimulus_metrics to be populated
        """
        if not self.best_params_per_stimulus or not self.stimulus_metrics:
            return
        
        # Prepare data for the table
        table_data = []
        for stim in self.best_params_per_stimulus.keys():
            fold_params = self.best_params_per_stimulus[stim]
            metrics = self.stimulus_metrics.get(stim, {})
            
            # Calculate most common parameters for this stimulus
            param_counts = {}
            for fold_info in fold_params:
                params = fold_info['params']
                param_key = (params['n_neighbors'], params['weights'], params['metric'])
                param_counts[param_key] = param_counts.get(param_key, 0) + 1
            
            # Get most common combination and its frequency
            most_common_params = max(param_counts, key=param_counts.get)
            frequency = param_counts[most_common_params]
            total_folds = len(fold_params)
            
            table_data.append({
                'Stimulus': stim,
                'k_neighbors': most_common_params[0],
                'weights': most_common_params[1],
                'metric': most_common_params[2],
                'param_frequency': frequency,
                'total_folds': total_folds,
                'param_consistency': frequency / total_folds,
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0)
            })
        
        # Create DataFrame and save
        params_table = pd.DataFrame(table_data)
        params_table = params_table.sort_values('accuracy', ascending=False)
        
        # Save detailed table
        table_file = self.viz_dir / 'parameter_performance_table.csv'
        params_table.to_csv(table_file, index=False, float_format='%.4f')
        print(f"Detailed parameter-performance table saved to: {table_file}")
        
        # Create a visualization of parameter vs performance
        self.create_parameter_performance_plot(params_table)
    
    def create_parameter_performance_plot(self, params_table):
        """
        Create scatter plots showing relationship between parameters and performance.
        
        Args:
            params_table (DataFrame): Table with parameters and performance metrics

        Output Files:
            Saves PNG image: {viz_dir}/parameter_performance_analysis.png
            Contains 2x2 subplot figure (15x12 inches, 300 DPI):
            * Accuracy vs Number of Neighbors (scatter plot with trend line)
            * F1 Score vs Number of Neighbors (scatter plot with trend line)
            * Accuracy distribution by Weight Type (box plots)
            * Accuracy distribution by Distance Metric (box plots)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. k vs Accuracy
        k_values = params_table['k_neighbors']
        accuracies = params_table['accuracy']
        scatter1 = ax1.scatter(k_values, accuracies, c='blue', alpha=0.6, s=100)
        ax1.set_xlabel('Number of Neighbors (k)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Number of Neighbors')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add trend line
        z1 = np.polyfit(k_values, accuracies, 1)
        p1 = np.poly1d(z1)
        ax1.plot(sorted(k_values), p1(sorted(k_values)), "r--", alpha=0.7)
        
        # 2. k vs F1 Score
        f1_scores = params_table['f1_score']
        scatter2 = ax2.scatter(k_values, f1_scores, c='red', alpha=0.6, s=100)
        ax2.set_xlabel('Number of Neighbors (k)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score vs Number of Neighbors')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add trend line
        z2 = np.polyfit(k_values, f1_scores, 1)
        p2 = np.poly1d(z2)
        ax2.plot(sorted(k_values), p2(sorted(k_values)), "r--", alpha=0.7)
        
        # 3. Weights vs Performance (box plots)
        weights_unique = params_table['weights'].unique()
        acc_by_weights = [params_table[params_table['weights'] == w]['accuracy'].values for w in weights_unique]
        bp1 = ax3.boxplot(acc_by_weights, labels=weights_unique, patch_artist=True)
        ax3.set_xlabel('Weight Type')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Distribution by Weight Type')
        ax3.grid(True, alpha=0.3)
        
        # Color the box plots
        colors = ['lightblue', 'teal']
        for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
            patch.set_facecolor(color)
        
        # 4. Distance Metric vs Performance (box plots)
        metrics_unique = params_table['metric'].unique()
        acc_by_metrics = [params_table[params_table['metric'] == m]['accuracy'].values for m in metrics_unique]
        bp2 = ax4.boxplot(acc_by_metrics, labels=metrics_unique, patch_artist=True)
        ax4.set_xlabel('Distance Metric')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy Distribution by Distance Metric')
        ax4.grid(True, alpha=0.3)
        
        # Color the box plots
        colors = ['turquoise', 'darkgreen']
        for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'parameter_performance_analysis.png', dpi=300, bbox_inches='tight')
        time.sleep(0.5)
        plt.close()
        
        print(f"Parameter-performance analysis saved to: {self.viz_dir / 'parameter_performance_analysis.png'}")
    
    def create_gaze_density_plots(self, df):
        """
        Create density plots for gaze data for each stimulus and class.
        
        Args:
            df (DataFrame): Original gaze data with X, Y coordinates

        Output Files:
            - Creates subdirectory: {viz_dir}/density_plots/
            - Saves multiple PNG images: {viz_dir}/density_plots/{stimulus}_density_{class}.png
            One plot per stimulus-class combination (e.g., "Stimulus1_density_C.png")
            Each plot: 10x8 inch hexbin density plot at 300 DPI
            * Class C (Controls): Blue colormap
            * Class P (Positive): Orange colormap
            * Inverted Y-axis for screen coordinates
            * Colorbar showing density scale
    
        Note:
            Only creates plots if create_visualizations=True in pipeline initialization
            Skips stimulus-class combinations with no data
        """
        if not self.create_visualizations:
            return
            
        print("Creating gaze density plots...")
        
        # Create density plots subfolder
        density_dir = self.viz_dir / "density_plots"
        density_dir.mkdir(exist_ok=True)
        
        # Get unique stimuli
        stimuli = df['Stimulus'].unique()
        
        # Loop through each stimulus and create separate plots for each class
        for stimulus in stimuli:
            # Filter data for the specific stimulus
            stimulus_data = df[df['Stimulus'] == stimulus]
            
            # Create plots for each class (C or P)
            for class_label in ['C', 'P']:
                # Filter data for the specific class
                class_data = stimulus_data[stimulus_data['Class'] == class_label]
                
                if len(class_data) == 0:
                    continue  # Skip if no data for this class
                
                # Create a new figure for the density plot
                plt.figure(figsize=(10, 8))
                
                # Choose different colormap for each class
                if class_label == 'C':
                    cmap = 'Blues'  # Use 'Blues' for Class C (Controls)
                elif class_label == 'P':
                    cmap = 'Oranges'  # Use 'Oranges' for Class P (Positive)
                
                # Use hexbin to represent density (points within each hexagon)
                plt.hexbin(
                    class_data['X'], 
                    class_data['Y'], 
                    gridsize=30,    # Size of the hexagonal grid
                    cmap=cmap,      # Use the class-specific colormap
                    alpha=0.6,      # Transparency for overlapping points
                )
                
                # Add colorbar to show density scale
                plt.colorbar(label='Density')
                
                # Invert Y axis (screen coordinate systems often have Y going down)
                plt.gca().invert_yaxis()
                
                # Add labels, title, and grid
                class_name = "Controls" if class_label == 'C' else "Positive"
                plt.title(f"Gaze Density Plot - {stimulus}\nClass: {class_name} ({class_label})")
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.grid(True, alpha=0.3)
                
                # Create safe filename
                safe_stimulus = "".join(c for c in stimulus if c.isalnum() or c in (' ', '-', '_')).rstrip()
                filename = density_dir / f"{safe_stimulus}_density_{class_label}.png"
                
                plt.tight_layout()
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                time.sleep(0.5)
                plt.close()
        
        print(f"Gaze density plots saved to: {density_dir}/")
    
    def run_knn_classification(self, input_path, output_file=None):
        """
        Run KNN classification on all stimuli and save results.
        
        Args:
            input_path (str): Path to the input CSV file
            output_file (str): Path for the output CSV file
        
        Returns:
            str: Path to the output file

        Output Files:
            Primary Output:
            - CSV file: {output_file} (default: "Final_Child_Predictions(GridSearch-KNN).csv" or "Final_Child_Predictions(5NN-D).csv")
            Contains Child ID and prediction_{stimulus_name} columns with probability values
            NaN values filled with 0.5 before saving
        
            Optional Parameters File (if use_gridsearch=True):
            - CSV file: "Best_Parameters_{output_file_base}.csv"
            Contains best hyperparameters found for each stimulus
        
            Visualization Files (if create_visualizations=True):
            - Individual confusion matrices: {viz_dir}/confusion_matrix_{stimulus}.png
            - Summary plots: {viz_dir}/accuracy_summary.png
            - Parameter analysis (if grid search): {viz_dir}/parameter_distributions.png, 
            {viz_dir}/parameter_performance_analysis.png, {viz_dir}/parameter_performance_table.csv
            - Gaze density plots: {viz_dir}/density_plots/{stimulus}_density_{class}.png
    
        Data Processing:
            - Performs leave-one-subject-out cross-validation
            - Optional hyperparameter optimization using nested cross-validation
            - Merges predictions from all stimuli into single output file
        """
        print("=" * 60)
        print("PHASE 1: KNN CLASSIFICATION")
        print("=" * 60)
        
        # Set default output file if not provided
        if not output_file:
            base_name = "Final_Child_Predictions"
            method = "GridSearch-KNN" if self.use_gridsearch else "5NN-D"
            output_file = f"{base_name}({method}).csv"
        
        print(f"Loading data from: {input_path}")
        
        # Load gaze data
        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"Error: File not found at {input_path}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        # Create density plots first
        self.create_gaze_density_plots(df)

        # Get unique stimuli names
        unique_stimuli = df["Stimulus"].unique()
        print(f"Found {len(unique_stimuli)} unique stimuli")
        
        # Initialize final results DataFrame
        final_results_df = pd.DataFrame()
        
        # Process each stimulus
        for stimulus_name in unique_stimuli:
            # Filter data for current stimulus
            df_stim = df[df["Stimulus"] == stimulus_name].copy()

            print("Start processing stim " + str(stimulus_name)) #MGV
            
            # Process this stimulus
            stim_results, best_params = self.process_stimulus(df_stim, stimulus_name)
            
            # Store best parameters if grid search was used
            if self.use_gridsearch and best_params:
                self.best_params_per_stimulus[stimulus_name] = best_params
            
            # Merge into final results
            if final_results_df.empty:
                final_results_df = stim_results
            else:
                final_results_df = pd.merge(final_results_df, stim_results, on="Child ID", how="outer")
        
        # Fill NaN values with 0.5 before saving
        prediction_columns = [col for col in final_results_df.columns if col.startswith("prediction_")]
        final_results_df[prediction_columns] = final_results_df[prediction_columns].fillna(0.5)
        
        # Save the merged predictions
        final_results_df.to_csv(output_file, index=False)
        print(f"\nFinal merged predictions saved to: {output_file}")
        print("NaN values in prediction columns filled with 0.5")
        
        # Create summary visualizations
        if self.create_visualizations:
            self.create_accuracy_summary_plot()
            if self.use_gridsearch:
                self.create_parameter_distribution_plots()
            print(f"Individual stimulus confusion matrices saved to: {self.viz_dir}/")
            print(f"Gaze density plots saved to: {self.viz_dir}/density_plots/")
        
        
        # Note: Best_Parameters file removed - all parameter info now in parameter_performance_table.csv
        # which provides more comprehensive analysis including performance metrics
        
        self.knn_results = output_file
        return output_file
    
    def _plot_rf_confusion_matrix(self, confusion_matrix):
        """
        Create and save the aggregated confusion matrix visualization for the Random Forest evaluation.

        Args:
            confusion_matrix (array): The aggregated confusion matrix (2x2, rows=true, cols=predicted).

        Output Files:
            Saves PNG image: {viz_dir}/random_forest_confusion_matrix.png

            - 8x6 inch heatmap of the confusion matrix with annotated cell values
            - Axis labels: 'Predicted Class', 'True Class'
            - Class labels: 'Control (C)', 'Positive (P)'
            - Colorbar indicating count
            - Saved at 300 DPI with tight bounding box

        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Control (C)', 'Positive (P)'],
            yticklabels=['Control (C)', 'Positive (P)'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Random Forest Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'random_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_rf_summary(self, accuracy, f1_score):
        """
        Create and save a summary bar plot of accuracy and F1 score for Random Forest evaluation.

        Args:
            accuracy (float): Mean accuracy score from cross-validation.
            f1_score (float): Mean F1 score from cross-validation.

        Output Files:
            Saves PNG image: {viz_dir}/random_forest_summary.png

            - 6x5 inch bar plot with two bars: Accuracy and F1 Score
            - Value labels above each bar
            - Y-axis range: [0, 1]
            - Title: 'Random Forest Performance Summary'
            - Saved at 300 DPI with tight bounding box

        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        metrics = ['Accuracy', 'F1 Score']
        values = [accuracy, f1_score]
        bars = plt.bar(metrics, values, color=['skyblue', 'lightcoral'])
        plt.ylim(0, 1)
        plt.title('Random Forest Performance Summary')
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'random_forest_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_with_random_forest(self, predictions_csv=None):
        """
        Evaluate the KNN predictions using Random Forest cross-validation.
        
        Args:
            predictions_csv (str): Path to the CSV file with predictions

        Output Files:
            - Text file: random_forest_evaluation_results.txt
            Contains cross-validation results including accuracy, F1 score, and confusion matrix

        Side Effects:
            - Creates visualizations if create_visualizations=True:
            - Saves confusion matrix plot: {viz_dir}/random_forest_confusion_matrix.png
            - Saves accuracy and F1 score summary plot: {viz_dir}/random_forest_summary.png
        """
        print("\n" + "=" * 60)
        print("PHASE 2: RANDOM FOREST EVALUATION")
        print("=" * 60)
        
        # Use the KNN results if no specific file is provided
        if predictions_csv is None:
            predictions_csv = self.knn_results
        
        if predictions_csv is None:
            print("Error: No predictions file available. Run KNN classification first.")
            return
        
        # Prepare results file path
        if self.create_visualizations:
            results_file = self.viz_dir / "random_forest_evaluation_results.txt"
        else:
            results_file = Path("random_forest_evaluation_results.txt")
        
        # Store console output in a list to write to file
        output_lines = []
        
        print(f"Loading predictions from: {predictions_csv}")
        output_lines.append(f"Loading predictions from: {predictions_csv}")
        
        # Load the merged prediction probabilities
        try:
            df = pd.read_csv(predictions_csv)
        except FileNotFoundError:
            error_msg = f"Error: File not found at {predictions_csv}"
            print(error_msg)
            output_lines.append(error_msg)
            return
        except Exception as e:
            error_msg = f"Error loading predictions: {e}"
            print(error_msg)
            output_lines.append(error_msg)
            return
        
        # Extract true class from first letter of 'Child ID'
        df['Class'] = df['Child ID'].str[0]
        
        # Get feature columns (prediction probabilities)
        feature_columns = [col for col in df.columns if col.startswith("prediction_")]
        msg = f"Found {len(feature_columns)} prediction features"
        print(msg)
        output_lines.append(msg)
        
        X = df[feature_columns].values
        y = df["Class"].values
        
        # Choose cross-validation strategy
        if self.rf_use_loo:
            from sklearn.model_selection import LeaveOneOut
            cv = LeaveOneOut()
            cv_name = "Leave-One-Out"
            n_folds = len(y)
        else:
            cv = StratifiedKFold(n_splits=self.rf_cv_splits, shuffle=True, random_state=42)
            cv_name = f"{self.rf_cv_splits}-fold Stratified"
            n_folds = self.rf_cv_splits
        
        accuracies = []
        f1_scores = []
        confusion = np.zeros((2, 2), dtype=int)  # For labels 'C' and 'P'
        
        label_order = ['C', 'P']
        
        cv_msg = f"Performing {cv_name} cross-validation ({n_folds} folds)..."
        print(cv_msg)
        output_lines.append(cv_msg)
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = rf.predict(X_test)
            
            # Calculate metrics
            fold_accuracy = accuracy_score(y_test, y_pred)
            fold_f1 = f1_score(y_test, y_pred, pos_label='P', zero_division=0)
            
            accuracies.append(fold_accuracy)
            f1_scores.append(fold_f1)
            confusion += confusion_matrix(y_test, y_pred, labels=label_order)
            
            # Print progress for LOO (every 10 folds) or all folds for k-fold
            if not self.rf_use_loo or fold % 10 == 0 or fold == n_folds:
                fold_msg = f"  Fold {fold}/{n_folds}: Accuracy={fold_accuracy:.4f}, F1={fold_f1:.4f}"
                print(fold_msg)
                output_lines.append(fold_msg)
        
        # Store results
        self.rf_results = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'confusion_matrix': confusion,
            'cv_method': cv_name
        }
        
        # Prepare final results summary
        output_lines.append(f"\nRandom Forest {cv_name} Cross-Validation Results:")
        output_lines.append(f"  - Average Accuracy: {self.rf_results['mean_accuracy']:.4f} (±{self.rf_results['std_accuracy']:.4f})")
        output_lines.append(f"  - Average F1 Score: {self.rf_results['mean_f1']:.4f} (±{self.rf_results['std_f1']:.4f})")
        output_lines.append("  - Aggregated Confusion Matrix (rows=true, cols=predicted):")
        output_lines.append("      C     P")
        output_lines.append(f"C  [{confusion[0,0]:3d}  {confusion[0,1]:3d}]")
        output_lines.append(f"P  [{confusion[1,0]:3d}  {confusion[1,1]:3d}]")
        
        # Print final results to console
        print(f"\nRandom Forest {cv_name} Cross-Validation Results:")
        print(f"  - Average Accuracy: {self.rf_results['mean_accuracy']:.4f} (±{self.rf_results['std_accuracy']:.4f})")
        print(f"  - Average F1 Score: {self.rf_results['mean_f1']:.4f} (±{self.rf_results['std_f1']:.4f})")
        print("  - Aggregated Confusion Matrix (rows=true, cols=predicted):")
        print("      C     P")
        print(f"C  [{confusion[0,0]:3d}  {confusion[0,1]:3d}]")
        print(f"P  [{confusion[1,0]:3d}  {confusion[1,1]:3d}]")
        
        # Write all output to results file
        try:
            with open(results_file, 'w') as f:
                f.write("RANDOM FOREST EVALUATION RESULTS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for line in output_lines:
                    f.write(line + "\n")
            
            print(f"\nRandom Forest evaluation results saved to: {results_file}")
            
        except Exception as e:
            print(f"Warning: Could not save results file: {e}")

        # Create visualizations if enabled
        if self.create_visualizations:
            self._plot_rf_confusion_matrix(self.rf_results['confusion_matrix'])
            self._plot_rf_summary(self.rf_results['mean_accuracy'], self.rf_results['mean_f1'])
            print(f"Confusion matrix plot saved to: {self.viz_dir / 'random_forest_confusion_matrix.png'}")
            print(f"Accuracy and F1 score summary plot saved to: {self.viz_dir / 'random_forest_summary.png'}")

    
    def run_complete_pipeline(self, input_path, output_file=None):
        """
        Run the complete pipeline: KNN classification followed by Random Forest evaluation.
        
        Args:
            input_path (str): Path to the input CSV file with gaze data
            output_file (str): Path for the KNN predictions output file
        
        Returns:
            dict: Dictionary containing both KNN and RF results
        Output Files:
            All files generated by run_knn_classification():
            - Primary predictions CSV
            - Optional best parameters CSV (if grid search enabled)
            - Visualization files (if enabled):
            * Individual confusion matrices
            * Summary accuracy plots
            * Parameter analysis plots and tables
            * Gaze density plots
        
        Processing Phases:
            1. KNN Classification: Generates prediction probabilities for each child-stimulus pair
            2. Random Forest Evaluation: Uses predictions as features for cross-validation assessment
        
        Pipeline Results:
            Returns dictionary with keys:
            - 'knn_predictions_file': Path to main predictions CSV
            - 'best_params': Dictionary of optimal parameters per stimulus
            - 'rf_results': Random Forest cross-validation metrics
        """
        print("Starting Complete Gaze Classification Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: KNN Classification
        predictions_file = self.run_knn_classification(input_path, output_file)
        
        if predictions_file is None:
            print("Pipeline failed during KNN classification phase.")
            return None
        
        # Phase 2: Random Forest Evaluation
        self.evaluate_with_random_forest(predictions_file)
        
        total_time = time.time() - start_time
        print(f"\nPipeline completed in {total_time:.2f} seconds")
        
        return {
            'knn_predictions_file': predictions_file,
            'best_params': self.best_params_per_stimulus,
            'rf_results': self.rf_results
        }

def main():
    """
    Main function to run the complete pipeline with default parameters.
    """
    # Configuration
    input_path = os.path.join("D:/GILLES/MATTIA/asdclassificationmattia-ongoing","merged_gaze_data.csv")  # Input gaze data file
    output_file = os.path.join("D:/GILLES/MATTIA/asdclassificationmattia-ongoing","output.csv")           # KNN predictions output
    use_gridsearch = True                # Whether to optimize KNN hyperparameters
    rf_cv_splits = 5                     # Number of folds for RF cross-validation (ignored if rf_use_loo=True)
    rf_use_loo = True                    # Whether to use leave-one-out CV for Random Forest
    create_visualizations = True         # Whether to create visualization plots
    
    # Initialize pipeline
    pipeline = GazeClassificationPipeline(
        use_gridsearch=use_gridsearch,
        rf_cv_splits=rf_cv_splits,
        create_visualizations=create_visualizations,
        rf_use_loo=rf_use_loo
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(input_path, output_file)
    
    if results:
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"✓ KNN predictions saved to: {results['knn_predictions_file']}")
        if results['best_params']:
            print(f"✓ Best parameters saved for {len(results['best_params'])} stimuli")
        if results['rf_results']:
            print(f"✓ Random Forest evaluation completed ({results['rf_results']['cv_method']})")
            print(f"  Final Accuracy: {results['rf_results']['mean_accuracy']:.4f}")
            print(f"  Final F1 Score: {results['rf_results']['mean_f1']:.4f}")
        if pipeline.create_visualizations:
            print(f"✓ Visualizations saved to: {pipeline.viz_dir}/")
            print("  - Individual confusion matrices for each stimulus")
            print("  - Summary accuracy plot for all stimuli")
            if pipeline.use_gridsearch:
                print("  - Parameter distribution analysis")
                print("  - Parameter-performance relationship plots")
                print("  - Detailed parameter-performance table")

if __name__ == "__main__":
    main()