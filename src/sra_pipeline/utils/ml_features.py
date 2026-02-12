"""
ML-ready feature table utilities for the SRA to Features Pipeline.
"""


from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import (
    RobustScaler, LabelBinarizer, LabelEncoder, label_binarize
)
from typing import List, Dict, Any, Optional

import itertools
import json
import logging
import matplotlib
matplotlib.use('Agg')
# Supress font messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import structlog

from ..models.features import FeatureSet


class FeatureTable:
    """Manage pipeline features and convert to tabular format."""

    def __init__(self, logger: structlog.BoundLogger):
        """
        Initialize feature table converter.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.feature_dicts: List[Dict] = []
    
    def add_feature_dict(self, feature_dict: Dict):
        """Add a feature set to the collection."""
        self.feature_dicts.append(feature_dict)
        self.logger.info("Added feature set", 
                         sra_id=feature_dict['sra_id'])
        return self

    def add_feature_dicts_from_directory(self, directory: Path):
        """
        Load feature dicts from a directory containing 
        features.json files.
        """
        self.logger.info("Loading feature dicts from directory", 
                         directory=str(directory))
        
        # Find all features.json files
        feature_files = list(directory.rglob("features.json"))
        
        for feature_file in feature_files:
            try:
                with open(feature_file, 'r') as f:
                    data = json.load(f)
                
                feature_dict = self._process_features(data)
                self.add_feature_dict(feature_dict)
                
            except Exception as e:
                self.logger.warning("Failed to load feature set", 
                                  file=str(feature_file), error=str(e))
        return self

    def create_sample_features_table(self) -> pd.DataFrame:
        """
        Create a samples × features table for ML training.
        
        Returns:
            DataFrame with samples as rows and features as columns
        """
        if not self.feature_dicts:
            raise ValueError("No feature dicts available")
        
        self.logger.info("Creating ML feature table", 
                         sample_count=len(self.feature_dicts))
        
        # Convert to DataFrame
        df = pd.DataFrame(self.feature_dicts)

        # Transpose table and set indices
        df = df.T
        df.columns = df.iloc[0]
        df = df[1:]

        # Remove rows that sum 0
        df = df[df.sum(axis=1)!=0]
        # Remove columns that sum 0
        df = df.loc[:, (df.sum(axis=0)!=0)]

        # Add 0 values to cnv features
        df.loc[df.index.astype(str).str.startswith("cnv_length")] = \
            df.loc[df.index.astype(str).str.startswith("cnv_length")].fillna(0)

        # Drop NA values so all samples have the same features
        df = df.dropna()

        # Reorder by columns
        df = df.sort_index(axis=1)

        # Reset index to make feature name a column
        df = df.reset_index()

        self.logger.info("ML feature table created", 
                        shape=df.shape, 
                        feature_count=len(df.columns))
        
        return df

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the feature table."""
        df = self.create_sample_features_table()
        
        return {
            'sample_count': len(df),
            'feature_count': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'feature_names': list(df.columns),
        }

    def save_feature_table(self, output_path: Path, format: str='csv'):
        """
        Save the feature table to file.
        
        Args:
            output_path: Output file path
            format: Output format ('csv', 'tsv', 'parquet', 'json')
        """
        # Get sample_features dataframe
        df: pd.DataFrame = self.create_sample_features_table()

        # Define if saving index is needed
        save_index = False

        # Make sure output_path exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in required format
        if format.lower() == 'csv':
            df.to_csv(output_path, index=save_index)
        elif format.lower() == 'tsv':
            df.to_csv(output_path, index=save_index, sep='\t')
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=save_index)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info("Feature table saved", 
                        path=str(output_path), 
                        format=format,
                        shape=df.shape)

        return self
    
    def _process_features(self, data) -> Dict:
        """
        Process raw feature data into a structured dictionary.
        
        Args:
            data: Raw feature data from JSON file
        
        Returns:
            Processed feature dictionary
        """
        # Initialize feature_dict
        feature_dict = {
            'sra_id': data.get('sra_id', '')
        }
        # Get different sub-elements from data
        genomic_bins: List[Dict] = data.get('genomic_bins', [])
        gene_stats: List[Dict] = data.get('gene_stats', [])
        cnv_regions: List[Dict] = data.get('cnv_regions', [])
        fragment_stats: Dict = data.get('fragment_stats', {})

        # Go through genomic_bins
        for i in range(len(genomic_bins)):
            curr_data = genomic_bins[i]
            chr_n = curr_data['chromosome']
            start = curr_data['start']
            end = curr_data['end']
            # Define key and value
            key = f'bin_gvs_{chr_n}:{start}-{end}'
            val = curr_data['variant_count']
            feature_dict[key] = int(val)
        
        # Go through gene_stats
        for i in range(len(gene_stats)):
            curr_data = gene_stats[i]
            gene_name = curr_data['gene_name'].split(':')[-1]
            start = curr_data['start']
            end = curr_data['end']
            total_gv = int(curr_data['total_variants'])
            dn_ds = curr_data['dn_ds_ratio']
            # Define key and value for GVs
            key_gv = f'gene_gvs_{gene_name}'
            feature_dict[key_gv] = int(total_gv)
            # Check if there are GVs to calculate dn/ds
            if total_gv!=0 and dn_ds is not None:
                key_dn_ds = f'gene_dn_ds_{gene_name}'
                feature_dict[key_dn_ds] = float(dn_ds)
        
        # Go through cnv_regions
        l_cnv_keys = []
        for i in range(len(cnv_regions)):
            curr_data = cnv_regions[i]
            chr_n = curr_data['chromosome']
            cnv_len = abs(int(curr_data['end']) - int(curr_data['start']))
            confidence = curr_data['confidence']
            copy_number = curr_data['copy_number']
            cnv_type = curr_data['type']
            # Filter by confidence
            if confidence > 0.999999:
                # Define key
                key = f'cnv_length_{cnv_type}_chr{chr_n}'
                # Make sure that key exists
                if not (key in feature_dict.keys()):
                    l_cnv_keys.append(str(key))
                    feature_dict[key] = 0
                # Define value to add
                if cnv_type=='gain':
                    copy_add = round(copy_number - 1)
                    val = cnv_len * max(copy_add, 1)
                elif cnv_type=='loss':
                    copy_loss = round(2.0 - (copy_number*2))
                    val = cnv_len * max(copy_loss, 1)
                else:
                    raise ValueError(f'Unrecognised cnv_type: {cnv_type}')
                # Add value to feature_dict[key]
                feature_dict[key] += val
        
        # Go through fragment_stats
        for key, val in fragment_stats.items():
            if key in ['max', 'min', 'mean', 'median', 'std']:
                feature_dict[f'fragment_length_{key}'] = float(val)
            else:
                feature_dict[f'fragment_{key}'] = float(val)
        
        return feature_dict


class MLFeatureTable:
    """Convert pipeline features to ML-ready tabular format."""
    
    def __init__(self, logger: structlog.BoundLogger):
        """
        Initialize ML feature table converter.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        self.feature_sets: List[FeatureSet] = []
    
    def add_feature_set(self, feature_set: FeatureSet):
        """Add a feature set to the collection."""
        self.feature_sets.append(feature_set)
        self.logger.info("Added feature set", sra_id=feature_set.sra_id)
    
    def add_feature_sets_from_directory(self, directory: Path):
        """Load feature sets from a directory containing features.json files."""
        self.logger.info("Loading feature sets from directory", directory=str(directory))
        
        # Find all features.json files
        feature_files = list(directory.rglob("features.json"))
        
        for feature_file in feature_files:
            try:
                with open(feature_file, 'r') as f:
                    data = json.load(f)
                
                feature_set = FeatureSet(**data)
                self.add_feature_set(feature_set)
                
            except Exception as e:
                self.logger.warning("Failed to load feature set", 
                                  file=str(feature_file), error=str(e))
    
    def create_sample_features_table(self) -> pd.DataFrame:
        """
        Create a samples × features table for ML training.
        
        Returns:
            DataFrame with samples as rows and features as columns
        """
        if not self.feature_sets:
            raise ValueError("No feature sets available")
        
        self.logger.info("Creating ML feature table", sample_count=len(self.feature_sets))
        
        # Initialize feature dictionary
        features_dict = {}
        
        for feature_set in self.feature_sets:
            sample_id = feature_set.sra_id
            
            # Basic sample features
            sample_features = {
                'sample_id': sample_id,
                'sample_name': feature_set.sample_name or sample_id,
                'pipeline_version': feature_set.pipeline_version,
                'processing_time': feature_set.processing_time,
            }
            
            # Quality metrics features
            qm = feature_set.quality_metrics
            quality_features = {
                'total_reads': qm.total_reads,
                'mapped_reads': qm.mapped_reads,
                'mapping_rate': qm.mapping_rate,
                'mean_coverage': qm.mean_coverage,
                'coverage_std': qm.coverage_std,
                'gc_content': qm.gc_content,
                'duplication_rate': qm.duplication_rate,
            }
            
            # Add quality scores
            for score_name, score_value in qm.quality_scores.items():
                quality_features[f'quality_score_{score_name}'] = score_value
            
            # Fragment length features
            fragment_features = {}
            if feature_set.fragment_stats:
                fs = feature_set.fragment_stats
                fragment_features = {
                    'fragment_mean': fs.mean,
                    'fragment_median': fs.median,
                    'fragment_std': fs.std,
                    'fragment_min': fs.min,
                    'fragment_max': fs.max,
                    'fragment_count': fs.count,
                }
            
            # Genomic variant features
            genomic_features = self._extract_genomic_features(feature_set.genomic_bins)
            
            # Gene-level features
            gene_features = self._extract_gene_features(feature_set.gene_stats)
            
            # CNV features
            cnv_features = self._extract_cnv_features(feature_set.cnv_regions)
            
            # Combine all features
            all_features = {
                **sample_features,
                **quality_features,
                **fragment_features,
                **genomic_features,
                **gene_features,
                **cnv_features,
            }
            
            features_dict[sample_id] = all_features
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(features_dict, orient='index')
        
        # Reset index to make sample_id a column
        df = df.reset_index(drop=True)
        
        self.logger.info("ML feature table created", 
                        shape=df.shape, 
                        feature_count=len(df.columns))
        
        return df
    
    def _extract_genomic_features(
        self,
        genomic_bins: List
    ) -> Dict[str, Any]:
        """Extract genomic variant features."""
        if not genomic_bins:
            return {}
        
        # Calculate summary statistics
        variant_counts = [bin.variant_count for bin in genomic_bins]
        
        return {
            'total_genomic_variants': sum(variant_counts),
            'mean_variants_per_bin': np.mean(variant_counts),
            'std_variants_per_bin': np.std(variant_counts),
            'max_variants_per_bin': max(variant_counts),
            'min_variants_per_bin': min(variant_counts),
            'bins_with_variants': sum(
                1 for count in variant_counts if count > 0
            ),
            'total_genomic_bins': len(genomic_bins),
            'variant_density': sum(variant_counts) / len(genomic_bins),
        }
    
    def _extract_gene_features(
        self,
        gene_stats: List
    ) -> Dict[str, Any]:
        """Extract gene-level variant features."""
        if not gene_stats:
            return {}
        
        # Calculate summary statistics
        total_variants = [gene.total_variants for gene in gene_stats]
        synonymous_variants = [gene.synonymous_variants \
                               for gene in gene_stats]
        nonsynonymous_variants = [gene.nonsynonymous_variants \
                                  for gene in gene_stats]
        dn_ds_ratios = [gene.dn_ds_ratio for gene in gene_stats \
                        if gene.dn_ds_ratio is not None]
        
        return {
            'total_genes_with_variants': len(gene_stats),
            'total_gene_variants': sum(total_variants),
            'mean_variants_per_gene': np.mean(total_variants),
            'std_variants_per_gene': np.std(total_variants),
            'total_synonymous_variants': sum(synonymous_variants),
            'total_nonsynonymous_variants': sum(nonsynonymous_variants),
            'mean_dn_ds_ratio': np.mean(dn_ds_ratios) \
                if dn_ds_ratios else 0,
            'std_dn_ds_ratio': np.std(dn_ds_ratios) \
                if dn_ds_ratios else 0,
            'genes_with_high_variants': sum(
                1 for count in total_variants if count > 10
            ),
        }
    
    def _extract_cnv_features(
        self,
        cnv_regions: List
    ) -> Dict[str, Any]:
        """Extract CNV features."""
        if not cnv_regions:
            return {}
        
        # Calculate summary statistics
        copy_numbers = [region.copy_number for region in cnv_regions]
        confidences = [region.confidence for region in cnv_regions]
        gains = sum(1 for region in cnv_regions \
                    if region.type == 'gain')
        losses = sum(1 for region in cnv_regions \
                     if region.type == 'loss')
        
        return {
            'total_cnv_regions': len(cnv_regions),
            'cnv_gains': gains,
            'cnv_losses': losses,
            'mean_copy_number': np.mean(copy_numbers),
            'std_copy_number': np.std(copy_numbers),
            'mean_cnv_confidence': np.mean(confidences),
            'high_confidence_cnvs': sum(
                1 for conf in confidences if conf > 0.9
            ),
            'cnv_burden': len(cnv_regions),
        }
    
    def save_feature_table(
        self,
        output_path: Path,
        format: str = 'csv'
    ):
        """
        Save the feature table to file.
        
        Args:
            output_path: Output file path
            format: Output format ('csv', 'tsv', 'parquet', 'json')
        """
        df = self.create_sample_features_table()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'tsv':
            df.to_csv(output_path, index=False, sep='\t')
        elif format.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info("Feature table saved", 
                        path=str(output_path), 
                        format=format,
                        shape=df.shape)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the feature table."""
        df = self.create_sample_features_table()
        
        return {
            'sample_count': len(df),
            'feature_count': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'feature_names': list(df.columns),
        }


def analyze_feature_groups_cvrf(
    df: pd.DataFrame,
    output_folder: Path,
    features_to_analyze: List[str],
    feature_n: int,
    target_column: str,
    logger: structlog.BoundLogger,
    split_n: int = 5,
    rand_seed: int | None = None
) -> pd.DataFrame:
    """
    Performs cross-validated Random Forest (RF) classification on all
    combinations of N features selected from a feature list.

    For each feature group, this function computes:
    - Mean AUC (macro, OVR)
    - Std AUC
    - Mean TPR (macro-averaged per class)
    - Mean FNR (macro-averaged per class)
    - Mean TNR (macro-averaged per class)
    - Mean FPR (macro-averaged per class)

    Args:
        df (pd.DataFrame): Input dataframe containing the features and
                           the target variable.
        output_folder (Path): Directory where output files will be
                              written.
        features_to_analyze (List[str]): List of feature names to use
                                         when forming groups.
        feature_n (int): Number of features per group.
        target_column (str): Name of the target variable column.
        logger (structlog.BoundLogger): Logger instance.
        split_n (int): Number of CV folds.
        rand_seed (int | None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A DataFrame with each row representing a feature
                      group and columns containing the computed
                      performance metrics.
    """

    # Check minimum feature count
    if len(features_to_analyze) < feature_n:
        msg = f"Please provide at least {feature_n} features for grouping."
        logger.error(msg, feature_n=feature_n)
        return pd.DataFrame()

    # Ensure output directories exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define X and y
    X = df[features_to_analyze]
    y = df[target_column]

    # Encode target labels once
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    # CV setup
    cv = StratifiedKFold(
        n_splits=split_n, shuffle=True, random_state=rand_seed
    )

    # Base model
    rf_base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=rand_seed,
        n_jobs=-1
    )

    # Generate all feature groups
    feature_groups = list(itertools.combinations(features_to_analyze, feature_n))
    logger.info(
        f"Generated {len(feature_groups)} feature groups from "
        f"{len(features_to_analyze)} features.",
        feature_n=feature_n
    )

    # Results
    results = {}
    n_groups = len(feature_groups)
    display_every = max(10, n_groups // 10)

    # Iterate through groups
    for i, curr_group in enumerate(feature_groups):
        group_id = f"ID_{i}"
        X_group = X[list(curr_group)].values

        auc_scores = []
        fold_tpr = []
        fold_fnr = []
        fold_tnr = []
        fold_fpr = []

        # Cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_group, y_encoded)):
            X_train = X_group[train_idx]
            X_test = X_group[test_idx]
            y_train = y_encoded[train_idx]
            y_test = y_encoded[test_idx]

            rf_model = clone(rf_base_model)

            try:
                # Train
                rf_model.fit(X_train, y_train)

                # Predict probabilities
                y_proba = rf_model.predict_proba(X_test)
                y_pred = rf_model.predict(X_test)

                # AUC
                auc_val = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(n_classes)
                )
                auc_scores.append(auc_val)

                # Confusion matrix
                cm = confusion_matrix(
                    y_test, y_pred, labels=np.arange(n_classes)
                )

                # Compute per-class rates
                tpr_list = []
                fnr_list = []
                tnr_list = []
                fpr_list = []

                for c in range(n_classes):
                    TP = cm[c, c]
                    FN = cm[c, :].sum() - TP
                    FP = cm[:, c].sum() - TP
                    TN = cm.sum() - (TP + FP + FN)

                    # TPR / FNR
                    if TP + FN > 0:
                        tpr_list.append(TP / (TP + FN))
                        fnr_list.append(FN / (TP + FN))

                    # TNR / FPR
                    if TN + FP > 0:
                        tnr_list.append(TN / (TN + FP))
                        fpr_list.append(FP / (TN + FP))

                # Per-fold mean across classes
                if tpr_list:
                    fold_tpr.append(np.mean(tpr_list))
                    fold_fnr.append(np.mean(fnr_list))
                if tnr_list:
                    fold_tnr.append(np.mean(tnr_list))
                    fold_fpr.append(np.mean(fpr_list))

            except ValueError as e:
                logger.warning(
                    f"Skipping fold {fold} for group {group_id} "
                    f"due to error: {e}"
                )

        # If nothing worked for this group
        if not auc_scores:
            results[group_id] = list(curr_group) + [
                np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan
            ]
            continue

        # Aggregate metrics
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)

        mean_tpr = np.mean(fold_tpr) if fold_tpr else np.nan
        mean_fnr = np.mean(fold_fnr) if fold_fnr else np.nan
        mean_tnr = np.mean(fold_tnr) if fold_tnr else np.nan
        mean_fpr = np.mean(fold_fpr) if fold_fpr else np.nan

        # Store results
        results[group_id] = (
            list(curr_group)
            + [
                mean_auc,
                std_auc,
                mean_tpr,
                mean_fnr,
                mean_tnr,
                mean_fpr
            ]
        )

        # Progress report
        if i == 0 or (i + 1) % display_every == 0:
            logger.debug(f"Progress: {i+1} / {n_groups}")

    # Build output DataFrame
    feature_cols = [f"Feature {i+1}" for i in range(feature_n)]
    metric_cols = [
        "AUC Mean", "AUC Std",
        "Mean TPR", "Mean FNR",
        "Mean TNR", "Mean FPR"
    ]

    output_df = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=feature_cols + metric_cols
    )

    output_df.index.name = "Feature Group ID"
    output_df = output_df.sort_values(by="AUC Mean", ascending=False)

    return output_df


def analyze_feature_pairs(
    df: pd.DataFrame,
    output_folder: Path,
    features_to_analyze: List[str],
    target_column: str,
    logger: structlog.BoundLogger,
    rand_seed: int=None
) -> pd.DataFrame:
    """
    Performs a classification analysis for all combinations of two 
    features from a given list and ranks them by AUC score.

    Args:
        df (pd.DataFrame): The DataFrame containing features and the 
                           target.
        output_folder (Path): Folder where output files are created.
        features_to_analyze (list): A list of feature names to be 
                                    paired.
        target_column (str): The name of the target variable column.
        logger: Logger instance.
        rand_seed (int): Random seed for repeatability.

    Returns:
        pd.DataFrame: A DataFrame of feature pairs ranked by AUC.
    """
    if len(features_to_analyze) < 2:
        logger.error(
            "Please provide at least two features for pairing.",
            feature_n=len(features_to_analyze)
        )
        return pd.DataFrame()

    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    X = df[features_to_analyze]
    y = df[target_column]

    # Generate all unique pairs of the selected features
    feature_pairs = list(itertools.combinations(
        features_to_analyze, 2
    ))
    logger.info(
        str(f"Generated {len(feature_pairs)} unique feature pairs "+\
            f"from the list of {len(features_to_analyze)} features."),
        feature_n=len(features_to_analyze)
    )

    pair_results = {}
    model = RandomForestClassifier(random_state=rand_seed)
    lb = LabelBinarizer()

    # Counter
    cont = 0
    ### Display
    n_feat_pairs = len(feature_pairs)
    ###

    for f1, f2 in feature_pairs:
        # Create a new DataFrame with only two features for training
        X_pair = X[[f1, f2]]

        X_train, X_test, y_train, y_test = train_test_split(
            X_pair,
            y,
            test_size=0.3,
            random_state=rand_seed,
            stratify=y
        )

        # Save train and test sets
        X_train.to_csv(f'{output_folder}/X_train_{cont}.csv')
        X_test.to_csv(f'{output_folder}/X_test_{cont}.csv')
        y_train.to_csv(f'{output_folder}/y_train_{cont}.csv')
        y_test.to_csv(f'{output_folder}/y_test_{cont}.csv')

        # Train a model
        trained_model = model.fit(X_train, y_train)

        # Pickle trained model
        pickle_model(trained_model,
                     f'{output_folder}/model_pair_ID_{cont}.pickle')

        # Handle multi-class classification for AUC calculation
        y_test_bin = lb.fit_transform(y_test)
        y_pred_proba = trained_model.predict_proba(X_test)

        if y_test_bin.shape[1] == 1:
            auc_score = roc_auc_score(y_test_bin, y_pred_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test_bin, y_pred_proba,
                                      multi_class='ovr')

        # Load the data
        pair_results[f'ID_{cont}'] = (f1, f2, auc_score)

        ### Display
        if cont==0 or (cont+1)%10==0:
            logger.debug(f'Progress: {cont+1} / {n_feat_pairs}',
                         feature_n=len(features_to_analyze))
        ###

        cont += 1
    
    # Create a DataFrame and sort the results
    pair_auc_df = pd.DataFrame.from_dict(
        pair_results, orient='index', columns=[
            'Feature 1', 'Feature 2', 'AUC Score'
        ]
    )
    pair_auc_df.index.name = 'Feature Pair ID'
    pair_auc_df = pair_auc_df.sort_values(by='AUC Score',
                                          ascending=False)

    return pair_auc_df


def analyze_feature_trios(
    df: pd.DataFrame,
    output_folder: Path,
    features_to_analyze: List[str],
    target_column: str,
    logger: structlog.BoundLogger,
    rand_seed: int=None
) -> pd.DataFrame:
    """
    Performs a classification analysis for all combinations of three 
    features from a given list and ranks them by AUC score.

    Args:
        df (pd.DataFrame): The DataFrame containing features and the 
                           target.
        output_folder (Path): Folder where output files are created.
        features_to_analyze (list): A list of feature names to be 
                                    grouped.
        target_column (str): The name of the target variable column.
        logger: Logger instance.
        rand_seed (int): Random seed for repeatability.

    Returns:
        pd.DataFrame: A DataFrame of feature pairs ranked by AUC.
    """
    if len(features_to_analyze) < 3:
        logger.error(
            "Please provide at least three features for pairing.",
            feature_n=len(features_to_analyze)
        )
        return pd.DataFrame()

    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    X = df[features_to_analyze]
    y = df[target_column]

    # Generate all unique trios of the selected features
    feature_trios = list(itertools.combinations(
        features_to_analyze, 3
    ))
    logger.info(
        str(f"Generated {len(feature_trios)} unique feature trios "+\
            f"from the list of {len(features_to_analyze)} features."),
        feature_n=len(features_to_analyze)
    )

    trio_results = {}
    model = RandomForestClassifier(random_state=rand_seed)
    lb = LabelBinarizer()

    # Counter
    cont = 0
    ### Display
    n_feat_trios = len(feature_trios)
    ###

    for f1, f2, f3 in feature_trios:
        # Create a new DataFrame with only the features for training
        X_pair = X[[f1, f2, f3]]

        X_train, X_test, y_train, y_test = train_test_split(
            X_pair,
            y,
            test_size=0.3,
            random_state=rand_seed,
            stratify=y
        )

        # Save train and test sets
        X_train.to_csv(f'{output_folder}/X_train_{cont}.csv')
        X_test.to_csv(f'{output_folder}/X_test_{cont}.csv')
        y_train.to_csv(f'{output_folder}/y_train_{cont}.csv')
        y_test.to_csv(f'{output_folder}/y_test_{cont}.csv')

        # Train a model
        trained_model = model.fit(X_train, y_train)

        # Pickle trained model
        pickle_model(trained_model,
                     f'{output_folder}/model_trio_ID_{cont}.pickle')

        # Handle multi-class classification for AUC calculation
        y_test_bin = lb.fit_transform(y_test)
        y_pred_proba = trained_model.predict_proba(X_test)

        if y_test_bin.shape[1] == 1:
            auc_score = roc_auc_score(y_test_bin, y_pred_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test_bin, y_pred_proba, 
                                      multi_class='ovr')

        # Load the data
        trio_results[f'ID_{cont}'] = (f1, f2, f3, auc_score)

        ### Display
        if cont==0 or (cont+1)%100==0:
            logger.debug(f'Progress: {cont+1} / {n_feat_trios}',
                         feature_n=len(features_to_analyze))
        ###

        cont += 1
    
    # Create a DataFrame and sort the results
    trio_auc_df = pd.DataFrame.from_dict(
        trio_results, orient='index', columns=[
            'Feature 1', 'Feature 2', 'Feature 3', 'AUC Score'
        ]
    )
    trio_auc_df.index.name = 'Feature Trio ID'
    trio_auc_df = trio_auc_df.sort_values(by='AUC Score', 
                                          ascending=False)

    return trio_auc_df


def analyze_features_and_rank_by_auc(
    df: pd.DataFrame,
    output_folder: Path,
    target_var: str,
    model,
    logger: structlog.BoundLogger,
    rand_seed:int=None
) -> pd.DataFrame:
    """
    Performs feature-by-feature analysis and ranks them by AUC score.

    Args:
        df (pd.DataFrame): The DataFrame with features and a 
                           target variable column.
        output_folder (Path): Folder where output files are created.
        target_var (str): Target variable to be predicted with the 
                          feature classification.
        model: A sklearn model instance.
        logger: Logger instance.
        rand_seed (int): Random seed for repeatability.

    Returns:
        pd.DataFrame: A DataFrame of features ranked by AUC.
    """
    # Obtain X and y
    X = df.drop(target_var, axis=1)
    y = df[target_var]
    # Initialize results dictionary
    results = {}
    # Define a folder for feature training and testing sets
    out_sets = output_folder / 'feature_sets'
    # Make sure the folder exists
    out_sets.mkdir(parents=True, exist_ok=True)
    ### Display
    # Counter
    cont = 0
    n_col = len(X.columns)
    ###
    # Initialize max_auc_score with 0
    max_auc_score = 0.0
    for feature in X.columns:
        auc_score = get_single_feature_auc(X, y, out_sets, feature,
                                           model, rand_seed,
                                           max_auc_score)
        # Update max_auc_score if needed
        if auc_score > max_auc_score:
            max_auc_score = auc_score
        # Store the result
        results[feature] = auc_score
        ### Display
        if cont==0 or (cont+1)%1000==0:
            logger.debug(f'Progress: {cont+1} / {n_col}')
        cont += 1
        ###
    
    # Create a DataFrame to store and sort the results
    auc_df = pd.DataFrame.from_dict(
        results, orient='index', columns=['AUC Score']
    ).sort_values(by='AUC Score', ascending=False)
    
    return auc_df


def cleanup_empty_rows_cols(df: pd.DataFrame):
    """
    Performs cleanup of a DataFrame by removing empty rows
    and columns.
    """
    # Replace empty strings with NaN
    df_out = df.replace('', np.nan)
    # Remove columns with NAN
    df_out = df_out.dropna(axis=1, how='any')
    # Remove rows that sum 0
    df_out = df_out[df.sum(axis=1)!=0]
    # Remove columns that sum 0
    df_out = df_out.loc[:, (df_out.sum(axis=0)!=0)]

    return df_out


def collapse_probabilities(y_pred: np.ndarray) -> np.ndarray:
    """
    Collapse probability predictions into class labels.
    """
    # Select maximum value per row and set it to 1, the rest to 0
    for i in range(len(y_pred)):
        curr_row = y_pred[i]
        max_val = 0
        for j in range(len(curr_row)):
            curr_cell = curr_row[j]
            if curr_cell>max_val:
                max_val = curr_cell
                max_col = int(j)
        for j in range(len(curr_row)):
            if j==max_col:
                y_pred[i][j] = 1
            else:
                y_pred[i][j] = 0
    # Set type to int
    y_pred = y_pred.astype(int)
    return y_pred


def create_auc_roc_curves(
    y_true_bin: np.ndarray,
    y_pred: np.ndarray,
    out_folder: Path,
    bname: str,
    out_name: str,
    model,
    logger: structlog.BoundLogger
) -> None:
    """
    Creates a ROC curve plot for multi-class classification.
    """

    logger.info('Calculating micro-averaged AUC score for OvR strategy.')

    # Prepare for plotting the one-vs-rest ROC curves
    n_classes = len(model.classes_)
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Plot chance level (diagonal line)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="navy", 
             label="Chance level", alpha=0.8)

    # Loop through each class to plot its individual ROC curve
    for i in range(n_classes):
        # Create a display for each class using RocCurveDisplay
        RocCurveDisplay.from_predictions(
            y_true_bin[:, i],
            y_pred[:, i],
            name=f"ROC curve (class {i})",
            ax=ax,
        )

    out_name_spaces = out_name.replace('_', ' ').title()
    title = 'Receiver Operating Characteristic (ROC) Curve '+\
        f'for {out_name_spaces}'
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(out_folder / f'roc_curve_{bname}{out_name}.png')


def create_conf_matrix(y_true, y_pred, classes, out_path):
    """
    Creates and saves a confusion matrix plot.
    """
    # Define integer labels
    int_labels = np.arange(len(classes))

    # Generate confusion matrix
    cmb = confusion_matrix(y_true, y_pred, labels=int_labels)

    fig, a0 = plt.subplots(figsize=(8,4))
    dispb = ConfusionMatrixDisplay(confusion_matrix=cmb, 
                                   display_labels=classes)
    out_name = out_path.stem.replace('_', ' ').title()
    a0.set_title(out_name)
    dispb.plot(ax=a0,colorbar=False)
    plt.tight_layout()
    fig.savefig(out_path)


def create_feature_table_from_directory(
    input_directory: Path,
    output_path: Path,
    logger: structlog.BoundLogger,
    format: str = 'csv'
) -> pd.DataFrame:
    """
    Create a feature table from a pipeline output directory.
    
    Args:
        input_directory: Directory containing pipeline results
        output_path: Output folder path for feature table
        format: Output format ('csv', 'tsv', 'parquet', 'json')
        logger: Logger instance
        
    Returns:
        DataFrame with samples × features
    """
    ml_table = FeatureTable(logger)
    ml_table.add_feature_dicts_from_directory(input_directory)
    
    if not ml_table.feature_dicts:
        raise ValueError(f"No feature sets found in {input_directory}")
    
    # Make sure output path is a directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Define output file path
    output_file = output_path / f'features_table.{format}'

    # Save feature table
    ml_table.save_feature_table(output_file, format)

    # Log summary
    summary = ml_table.get_feature_summary()
    logger.info("ML feature table created successfully", **summary)
    
    return ml_table.create_sample_features_table()


def create_heatmap(
    df:pd.DataFrame,
    title:str,
    save_fig:bool,
    out_path:str,
    logger: structlog.BoundLogger
) -> None:
    """Creates a heatmap using Seaborn and Matplotlib"""
    # Check that df is not empty
    if df.empty:
        logger.warning(
            f'Heatmap with title "{title}" could not be created.'
        )
        return None
    logger.info(f'Creating heatmap with title "{title}"...',
                title=title)
    # Define the figure size
    fig_width = 9
    fig_height = 9
    # Set the font scale to make labels readable
    sns.set_theme(font_scale=0.8)
    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Load the dataframe
    sns.heatmap(df, cmap="viridis", ax=ax, vmin=0) # vmin=-20, vmax=20
    # Add titles and formatting
    plt.title(f"Features Heatmap for {title}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Show or save the plot
    if save_fig:
        path_out = os.path.join(out_path, f'heatmap_{title}.png')
        plt.savefig(path_out)
    return None


def create_ml_feature_table_from_directory(
    input_directory: Path,
    output_path: Path,
    logger: structlog.BoundLogger,
    format: str = 'csv'
) -> pd.DataFrame:
    """
    Create ML-ready feature table from pipeline output directory.
    
    Args:
        input_directory: Directory containing pipeline results
        output_path: Output file path for feature table
        format: Output format ('csv', 'tsv', 'parquet', 'json')
        logger: Logger instance
        
    Returns:
        DataFrame with samples × features
    """
    ml_table = MLFeatureTable(logger)
    ml_table.add_feature_sets_from_directory(input_directory)
    
    if not ml_table.feature_sets:
        raise ValueError(f"No feature sets found in {input_directory}")
    
    # Save feature table
    ml_table.save_feature_table(output_path, format)
    
    # Log summary
    summary = ml_table.get_feature_summary()
    logger.info("ML feature table created successfully", **summary)
    
    return ml_table.create_sample_features_table()


def create_macro_avg_auc_roc_curve(
    y_true_bin: np.ndarray,
    y_pred: np.ndarray,
    out_folder: Path,
    bname: str,
    out_name: str,
    logger: structlog.BoundLogger
) -> None:
    """
    Creates a macro-averaged ROC curve plot for multi-class 
    classification.
    
    Args:
        y_true_bin (np.ndarray): True binary labels (one-hot encoded).
        y_pred (np.ndarray): Predicted probabilities 
                             (n_samples, n_classes).
        out_folder (Path): Folder to save the plot.
        bname (str): Base name for the output file.
        out_name (str): Additional name component for the output and 
                        plot title.
        logger (structlog.BoundLogger): Logger instance.
    """

    logger.info('Creating macro-averaged ROC curve.')
    
    # Determine the number of classes
    n_classes = y_true_bin.shape[1]

    # Dictionaries to store per-class ROC components
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate ROC curve and AUC for each class
    for i in range(n_classes):
        # Calculate the ROC curve for class 'i' as the positive class 
        # against all others
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates (FPRs)
    all_fpr = np.unique(np.concatenate(
        [fpr[i] for i in range(n_classes)]
    ))

    # Interpolate all ROC curves at these common FPR points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        # Interpolate the TPR for class 'i' to match the 'all_fpr' points
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Average the interpolated TPRs
    mean_tpr /= n_classes

    # Store the final macro-average results
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    macro_roc_auc = auc(fpr["macro"], tpr["macro"])

    # Generate and save the AUC graph
    plt.figure(figsize=(8, 6))

    # Plot the macro-average curve
    label_str = f'Macro-average ROC curve (AUC = {macro_roc_auc:.2f})'
    plt.plot(fpr["macro"], tpr["macro"], label=label_str, color='blue',
             linewidth=2)
             
    # Plot the chance level line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Chance level')

    # Define title of the graph
    out_name_spaces = out_name.replace('_', ' ').lstrip().title()
    title = 'Macro-averaged Receiver Operating Characteristic (ROC) ' + \
        f'Curve for {out_name_spaces}'
    # Set title and labels of the graph
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(out_folder / f'macro_avg_roc_curve_{bname}{out_name}.png')


def create_micro_avg_auc_roc_curve(
    y_true_bin: np.ndarray,
    y_pred: np.ndarray,
    out_folder: Path,
    bname: str,
    out_name: str,
    logger: structlog.BoundLogger
) -> None:
    """
    Creates a micro-averaged ROC curve plot for multi-class 
    classification.

    Args:
        y_true_bin (np.ndarray): True binary labels (one-hot encoded).
        y_pred (np.ndarray): Predicted probabilities 
                             (n_samples, n_classes).
        out_folder (Path): Folder to save the plot.
        bname (str): Base name for the output file.
        out_name (str): Additional name component for the output and 
                        plot title.
        logger (structlog.BoundLogger): Logger instance.
    """

    logger.info('Creating micro-averaged ROC curve.')
    
    # Calculate micro-averaged ROC curve
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    micro_roc_auc = auc(fpr, tpr)

    # Generate and save the AUC graph
    plt.figure(figsize=(8, 6))

    # Plot the micro-average curve
    label_str = f'Micro-average ROC curve (AUC = {micro_roc_auc:.2f})'
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=label_str)

    # Plot the chance level line
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1),
             color='navy', lw=2, linestyle='--', label='Chance level')

    # Define title of the graph
    out_name_spaces = out_name.replace('_', ' ').lstrip().title()
    title = 'Micro-averaged Receiver Operating Characteristic (ROC) '+\
        f'Curve for {out_name_spaces}'
    # Set title and labels of the graph
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(out_folder / f'micro_avg_roc_curve_{bname}{out_name}.png')


def cross_validate_rf(
    df: pd.DataFrame,
    target_var: str,
    output_folder: Path,
    logger,
    rand_seed: int|None,
    top_n: int=0,
    refit: str='roc_auc'
) -> List[Any]:
    """
    Perform cross-validation with a Random Forest classifier.

    Args:
        df (pd.DataFrame): DataFrame containing features and target 
                           variable.
        target_var (str): Name of the target variable column.
        output_folder (Path): Folder where output files are created.
        logger: Logger instance.
        rand_seed (int): Random seed for repeatability.
        refit (str): Metric to optimize during hyperparameter tuning.
                     Default is 'roc_auc'. 
                     Recommended alternative is 'f1'. 

    Returns:
        float: Mean ROC-AUC score across folds.
        Any: Best model from cross-validation.
        List[str]: List of top N feature names by importance.
    """

    X = df.drop(columns=[target_var])
    y = df[target_var]

    model = RandomForestClassifier(random_state=rand_seed)
    
    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Store the mapping for reference
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    logger.info(f"Target label mapping: {label_mapping}")

    logger.info("Splitting data into training and testing sets...")

    # Split the data into training and testing sets
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=rand_seed
    )

    # Define hyperparameters for CV to search over
    cv_params = {
        'max_depth': [2,3],
        'min_samples_leaf': [1,2,3],
        'min_samples_split': [2,3,4],
        'max_features': [1,2,3,4],
        'n_estimators': [75, 100]
    }
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted', 
                                 zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', 
                              zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', 
                          zero_division=0),
        'roc_auc': 'roc_auc_ovr' 
                    # make_scorer(roc_auc_score, multi_class='ovr')
    }
    refit = 'roc_auc' # Metric to optimize

    logger.info("Instantiating cross-validated Random Forest model...")

    # Instantiate the cross validated random forest
    model_cv = GridSearchCV(model, cv_params, scoring=scoring,
                            cv=5, refit=refit, n_jobs=4)

    logger.info("Fitting model with cross-validation...")

    # Fit the model
    model_cv.fit(Xb_train, yb_train)

    logger.info("Model fitting completed.")

    logger.info("Getting best model from cross-validation...")

    # Get the best model and best parameters
    best_model = model_cv.best_estimator_
    best_params = model_cv.best_params_

    # Save best model
    pickle_model(best_model, 
                 f'{output_folder}/best_model_single.pickle')

    logger.info("Best model found", best_params=best_params)

    logger.info("Evaluating best model on test set...")

    # Evaluate the best model on the test set
    yb_pred_proba = best_model.predict_proba(Xb_test)

    logger.info("Model evaluation on test set completed.")

    logger.info("Calculating AUC score...")

    # Calculate mean score
    try:
        mean_score = roc_auc_score(yb_test, yb_pred_proba,
                                   multi_class='ovr')
    except Exception as e:
        logger.warning(f"Final ROC-AUC calculation failed: {e}")
        mean_score = 0.5

    logger.info("Cross-validation completed.", mean_auc=mean_score)

    # Get predictions on the test set
    yb_pred = best_model.predict(Xb_test)

    # Plot confusion matrix
    cmb = confusion_matrix(yb_test, yb_pred, 
                           labels=best_model.classes_)

    fig, a0 = plt.subplots(figsize=(8,4))
    dispb = ConfusionMatrixDisplay(confusion_matrix=cmb, 
                                   display_labels=best_model.classes_)
    a0.set_title("Best CVRF model (single feature)")
    dispb.plot(ax=a0,colorbar=False)
    plt.tight_layout()
    fig.savefig(
        f'{output_folder}/confusion_matrix.png'
    )

    importance = best_model.feature_importances_
    forest_importance = pd.Series(importance, 
                                  index=model_cv.feature_names_in_)

    # Get feature names - use the best model's feature names
    if hasattr(best_model, 'feature_names_in_'):
        feature_names = best_model.feature_names_in_
    else:
        # Fallback: use column names from the original DataFrame
        feature_names = X.columns.tolist()

    # Create a Series with feature importances and names
    forest_importance = pd.Series(importance, index=feature_names)

    # Sort by importance (descending)
    importance_sorted = forest_importance.sort_values(
        ascending=False
    )

    # Save the sorted importances to a CSV file
    importance_sorted.to_csv(
        os.path.join(output_folder, f'feature_importances.csv'),
        header=['Importance']
    )

    if top_n >= 2:
        # Select top n features by importance
        top_n_features = importance_sorted.head(top_n)

        # Plot the top n feature importances
        fig, ax = plt.subplots()
        top_n_features.plot.bar(ax=ax)
        ax.set_title(f"Top {top_n} feature importances")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.savefig(
            f'{output_folder}/top_{top_n}_feature_importances.png'
        )

        # Get just the names of the top n features as a list
        top_n_feature_names = top_n_features.index.tolist()

        # Print or log the results
        print("Top 20 features by importance:")
        for i, feature_name in enumerate(top_n_feature_names, 1):
            print(f"{i}. {feature_name}:"+\
                  f" {top_n_features[feature_name]:.6f}")
    else:
        top_n_feature_names = []

    return mean_score, best_model, top_n_feature_names


def cross_validated_feature_analysis(
    table_name: str,
    output_folder: Path,
    target_var: str,
    logger: structlog.BoundLogger,
    top_n: int=20,
    split_n: int=5,
    rand_seed: int=None
) -> None:
    """
    Performs RandomForest CrossValidated feature analysis.
    """
    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    # Define base name for outputs
    bname = str(table_name).rsplit('/')[-1].rsplit('.')[0]

    # Read feature table
    df = pd.read_csv(table_name, index_col=0)
    # Transpose df to have samples as rows and features as columns
    df = df.T

    # Perform CVRF
    features_metrics_df = top_single_features_cvrf(
        df,
        target_var,
        logger,
        split_n=split_n,
        random_state=rand_seed
    )
    # Save top features and their scores
    features_metrics_df.to_csv(output_folder / f"{bname}_top_features.csv",
                               header=['AUC Score', 'AUC Std', 'Mean FPR',
                                       'Mean FNR', 'Mean TPR', 'Mean TNR'])
    # Pick the top_n features and get the feature names
    top_features = features_metrics_df.head(top_n)
    top_feature_names = top_features.index

    # Train final models and store them
    final_models_and_results = train_and_store_final_models(
        df,
        target_var,
        output_folder / 'top_models',
        top_feature_names,
        top_features,
        logger,
        rand_seed
    )

    # Generate ROC curves for the top features
    logger.info("Generating ROC curves for top features")

    for feature_name, (mean_cv_auc, fitted_model, model_path, feature_list, full_dataset_metrics_dict) in \
        final_models_and_results.items():
        plot_feature_roc(
            df, 
            target_var, 
            output_folder / 'top_models',
            feature_list,
            fitted_model,
            logger,
            random_state=rand_seed
        )

    # Run in pairs/trios
    if top_n >= 2:

        logger.debug(f'Top {top_n} features obtained.',
                     top_features=top_feature_names)

        logger.info(f'Analyzing top {top_n} features in pairs.',
                    table_name=table_name,
                    top_n=top_n)

        # Define pairs output name
        output_folder_pairs = output_folder / "pairs_auc_out"
        output_pairs = output_folder_pairs / \
            f"{bname}_feature_pairs.csv"

        # Run the analysis
        top_feature_pairs_ranked = analyze_feature_groups_cvrf(
            df,
            output_folder_pairs,
            top_feature_names,
            2,
            target_var,
            logger,
            split_n=split_n,
            rand_seed=rand_seed
        )

        # Save the resulting dataframe
        top_feature_pairs_ranked.to_csv(output_pairs)

        # Pick the top_n groups of features and get their feature group IDs
        top_feature_metrics = top_feature_pairs_ranked.head(top_n)
        top_group_ids = top_feature_metrics.index.tolist()

        final_models_pairs = train_and_store_final_models_for_groups(
            df,
            target_var,
            output_folder_pairs / 'top_models',
            top_group_ids,
            top_feature_metrics,
            logger,
            random_state=rand_seed
        )

        for group_id, (mean_cv_auc, fitted_model, model_path, feature_list, full_dataset_metrics_dict) in \
            final_models_pairs.items():
            plot_feature_roc(
                df, 
                target_var, 
                output_folder_pairs / 'top_models',
                feature_list,
                fitted_model,
                logger,
                random_state=rand_seed
            )
        
        if top_n >= 3:
            logger.info(f'Analyzing top {top_n} features in trios.',
                    table_name=table_name,
                    top_n=top_n)
            # Define trios output name
            output_folder_trios = output_folder / "trios_auc_out"
            output_trios = output_folder_trios / \
                f"{bname}_feature_trios.csv"

            # Run the analysis for trios
            top_feature_trios_ranked = analyze_feature_groups_cvrf(
                df,
                output_folder_trios,
                top_feature_names,
                3,
                target_var,
                logger,
                split_n=split_n,
                rand_seed=rand_seed
            )

            # Save the resulting dataframe
            top_feature_trios_ranked.to_csv(output_trios)

            # Pick the top_n groups of features and get their feature group IDs
            top_feature_metrics = top_feature_trios_ranked.head(top_n)
            top_group_ids = top_feature_metrics.index.tolist()

            final_models_trios = train_and_store_final_models_for_groups(
                df,
                target_var,
                output_folder_trios / 'top_models',
                top_group_ids,
                top_feature_metrics,
                logger,
                random_state=rand_seed
            )

            for group_id, (mean_cv_auc, fitted_model, model_path, feature_list, full_dataset_metrics_dict) in \
                final_models_trios.items():
                plot_feature_roc(
                    df, 
                    target_var, 
                    output_folder_trios / 'top_models',
                    feature_list,
                    fitted_model,
                    logger,
                    random_state=rand_seed
                )
    return None


def cvrf_load_top_features(
    df: pd.DataFrame,
    data_table_path: Path,
    target_var: str,
    output_folder: Path,
    top_features_file: Path,
    bname: str,
    group_n: int,
    logger: structlog.BoundLogger,
    top_n: int=20,
    split_n: int=10,
    rand_seed: int=None
) -> None:
    """
    Runs the cross_validated_feature_analysis pipeline by loading
    top features from a previous run.
    """
    # Open top features file to get the ordered feature scores
    features_metrics_df = pd.read_csv(top_features_file, index_col=0)

    # Pick the top_n features and get the feature names
    top_features = features_metrics_df.head(top_n)
    top_feature_names = top_features.index

    # Train final models and store them
    final_models_and_results = train_and_store_final_models(
        df, 
        target_var, 
        output_folder / 'top_models',
        top_feature_names, 
        top_features,
        logger,
        rand_seed
    )

    # Generate ROC curves for the top features
    logger.info("Generating ROC curves for top features")
    for feature_name, (mean_cv_auc, fitted_model, model_path, feature_list, full_dataset_metrics_dict) in \
        final_models_and_results.items():
        #plot_feature_roc(
        #    df, 
        #    target_var, 
        #    output_folder / 'top_models',
        #    [feature_name],
        #    fitted_model,
        #    logger,
        #    random_state = rand_seed
        #)
        run_model_validation_and_test(
            model_path=model_path,
            data_table_path=data_table_path,
            out_folder=output_folder / 'top_models',
            target_var='Diagnosis',
            logger=logger,
            out_name='',
            feature_names=feature_list
        )

    # Run in pairs/trios
    if (group_n > 1) and (top_n >= group_n):

        logger.debug(f'Top {top_n} features obtained.',
                     top_features=top_feature_names)

        logger.info(
            f'Analyzing top {top_n} features in groups of {group_n}.',
            top_n=top_n
        )

        # Define group output name
        output_folder_group = output_folder / \
            f"{group_n}_groups_auc_out"
        output_group = output_folder_group / \
            f"{bname}_feature_{group_n}_groups.csv"

        # Run the analysis
        top_feature_groups_ranked = analyze_feature_groups_cvrf(
            df,
            output_folder_group,
            top_feature_names,
            group_n,
            target_var,
            logger,
            split_n=split_n,
            rand_seed=rand_seed
        )
        # Save the resulting dataframe
        top_feature_groups_ranked.to_csv(output_group)

    return None


def evaluate_model(
    model_path: Path,
    data_table_path: Path,
    out_folder: Path,
    out_name: str,
    logger: structlog.BoundLogger,
    target_var: str='Diagnosis',
    feature_list: list[str]=[],
    label_list: list[str]=['Healthy', 'BRC', 'CRC']
):
    """
    Evaluates a trained model on a data table and creates confusion 
    matrix plots.
    """

    logger.info('Evaluating model.', model_path=model_path,
                data_table_path=data_table_path,
                out_folder=out_folder,
                out_name=out_name,
                target_var=target_var,
                feature_list=feature_list)
    
    out_folder.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_pickled_model(model_path)

    # Load data
    df = pd.read_csv(data_table_path, sep=',', index_col=0)
    X = df.T.drop(columns=[target_var])
    y = df.T[target_var]

    if label_list:
        model_classes = label_list.copy()
    else:
        model_classes = list(model.classes_)

    # Last fallback: use sorted unique labels
    if any(lbl not in model_classes for lbl in y.unique()):
        model_classes = sorted(y.unique())

    # Mapping label to integer according to model_classes
    label_to_int = {label: i for i, label in enumerate(model_classes)}
    y_true_int = y.map(label_to_int).values

    # Select only model features
    if feature_list:
        model_features = feature_list
    else:
        model_features = model.feature_names_in_

    X = X[model_features]

    # Predict probabilities
    y_pred_proba = model.predict_proba(X)

    # Convert probs to integer labels
    y_pred_int = np.argmax(y_pred_proba, axis=1)

    # Ensure class label list length matches number of classes
    class_labels = label_list[:len(y_pred_proba[0])]
    # Define output confusion matrix path
    out_conf = out_folder / f'{out_name}.png'

    # Create confusion matrix
    create_conf_matrix(y_true_int, y_pred_int, class_labels, out_conf)

    # Generate ROC curves
    generate_roc_curves(
        y_true_int=y_true_int,
        y_pred_proba=y_pred_proba,
        class_labels=model_classes,
        out_folder=out_folder,
        out_name=out_name
    )

    return None


def full_evaluation(
    model_path: Path,
    X: pd.DataFrame,
    y_true: pd.Series,
    output_dir: Path,
    logger: structlog.BoundLogger,
    all_labels: Optional[List[str]] = None,
    pdf_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a pickled multiclass classifier.

    Args:
        model_path (Path): Path to a pickled model file.
        X (pd.DataFrame): Feature matrix (rows aligned with y_true).
        y_true (pd.Series): True labels for X.
        output_dir (Path): Directory where outputs will be saved. 
                           Will be created if missing.
        all_labels (list[str]): Complete, canonical order of labels. 
                                Default is ["Healthy", "CRC", "BRC"].
        pdf_title (str): If provided, included as title page on the 
                         PDF report. If None or "", PDF still created 
                         but without title text.
        logger (structlog.BoundLogger): Logger instance
    Returns:
        dict: A dictionary summarizing paths to generated files and 
              main metrics.
    """

    logger.info('Evaluating model.', model_path=model_path)

    # Prepare output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default label order
    if all_labels is None:
        label_order = ["Healthy", "CRC", "BRC"]
    else:
        label_order = list(all_labels)

    # Load model
    model = load_pickled_model(Path(model_path))

    # Check predict_proba availability
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model has no predict_proba(); ROC curves require probability outputs. Exiting as requested.")

    # Ensure X matches the model's expected feature names
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

        # Add missing columns as zeros
        for col in expected:
            if col not in X.columns:
                X[col] = 0

        # Drop extra columns not used by the model
        X = X[expected]

    # Get predicted labels and probabilities
    y_pred = pd.Series(model.predict(X), index=X.index)
    proba = model.predict_proba(X) # columns correspond to model.classes_
    model_classes = list(getattr(model, "classes_", []))

    # align probability columns to label_order
    # create proba_aligned shape (n_samples, n_labels)
    n_samples = proba.shape[0]
    n_labels = len(label_order)
    proba_aligned = np.zeros((n_samples, n_labels), dtype=float)

    # map model classes to column indices
    model_class_to_idx = {c: i for i, c in enumerate(model_classes)}
    for j, label in enumerate(label_order):
        if label in model_class_to_idx:
            proba_aligned[:, j] = proba[:, model_class_to_idx[label]]
        else:
            # label not present in model outputs: leave column zeros
            proba_aligned[:, j] = 0.0

    # Binarize y_true using the canonical label_order
    y_true_list = list(y_true)
    try:
        y_true_bin = label_binarize(y_true_list, classes=label_order)
    except Exception as e:
        # fallback: manual binarization
        y_true_bin = np.zeros((len(y_true_list), n_labels), dtype=int)
        for i, lab in enumerate(y_true_list):
            if lab in label_order:
                y_true_bin[i, label_order.index(lab)] = 1

    # Compute per-class ROC curves and AUC
    per_class_curves = {} # stores fpr,tpr,auc or None if not computable
    per_class_auc = {}
    for j, label in enumerate(label_order):
        # Check if label occurs in y_true
        positives = y_true_bin[:, j].sum()
        negatives = len(y_true_bin) - positives
        if positives == 0 or negatives == 0:
            # cannot compute ROC for this class (need both classes)
            per_class_curves[label] = None
            per_class_auc[label] = float("nan")
        else:
            fpr, tpr, _ = roc_curve(y_true_bin[:, j], 
                                    proba_aligned[:, j])
            try:
                a = auc(fpr, tpr)
            except Exception:
                a = float("nan")
            per_class_curves[label] = {"fpr": fpr, "tpr": tpr}
            per_class_auc[label] = float(a)

    # Compute macro and micro AUCs robustly:
    # macro: mean of available per-class AUCs (ignoring NaN)
    valid_aucs = [v for v in per_class_auc.values() if not (np.isnan(v))]
    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else float("nan")

    # micro: try sklearn roc_auc_score with multilabel indicator; if fails set nan
    micro_auc = float("nan")
    try:
        # requires at least one positive and one negative across all classes
        micro_auc = float(roc_auc_score(y_true_bin, proba_aligned, 
                                        average="micro"))
    except Exception:
        micro_auc = float("nan")

    # Build and save ROC plots
    roc_per_class_path = output_dir / "roc_per_class.png"
    roc_macro_path = output_dir / "roc_macro.png"
    roc_micro_path = output_dir / "roc_micro.png"

    # Per-class ROC plot (plot only classes with computable curves)
    plt.figure(figsize=(8, 6))
    any_plotted = False
    for label in label_order:
        curve = per_class_curves[label]
        if curve is not None:
            plt.plot(curve["fpr"], curve["tpr"], 
                     label=f"{label} (AUC={per_class_auc[label]:.3f})")
            any_plotted = True
        else:
            # skip plotting missing curves
            pass
    if any_plotted:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # diagonal
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Per-class ROC curves")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_per_class_path, dpi=150)
    else:
        # create an empty image indicating no per-class ROC available
        plt.text(0.5, 0.5, "No per-class ROC curves could be computed\n(need both positive and negative samples per class).",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(roc_per_class_path, dpi=150)
    plt.close()

    # Macro ROC: create a plot showing macro as mean curve if possible
    plt.figure(figsize=(8, 6))
    any_plotted = False
    # Plot per-class curves (if any)
    #for label in label_order:
    #    curve = per_class_curves[label]
    #    if curve is not None:
    #        plt.plot(curve["fpr"], curve["tpr"], lw=1, alpha=0.6, label=f"{label}")
    #        any_plotted = True
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Macro-average ROC (macro AUC = {macro_auc:.3f})")
    if any_plotted:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_macro_path, dpi=150)
    plt.close()

    # Micro ROC: compute aggregate micro ROC if possible
    # micro curve: flatten all true/score pairs
    micro_roc_path_exists = False
    plt.figure(figsize=(8, 6))
    try:
        # Only compute if roc_curve supports the flattened arrays (requires positive and negative)
        y_true_flat = y_true_bin.ravel()
        y_score_flat = proba_aligned.ravel()
        # only if y_true_flat contains both 0 and 1
        if (y_true_flat.sum() > 0) and \
            (y_true_flat.sum() < len(y_true_flat)):
            fpr_m, tpr_m, _ = roc_curve(y_true_flat, y_score_flat)
            auc_m = auc(fpr_m, tpr_m)
            plt.plot(fpr_m, tpr_m, 
                     label=f"micro-average (AUC={auc_m:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Micro-average ROC")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_micro_path, dpi=150)
            micro_roc_path_exists = True
    except Exception:
        micro_roc_path_exists = False

    if not micro_roc_path_exists:
        plt.text(0.5, 0.5, 
                 "Micro-average ROC could not be computed\n(need both positive and negative labels across classes).",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(roc_micro_path, dpi=150)
    plt.close()

    # Confusion matrix heatmap
    conf = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_path = output_dir / "confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf, interpolation="nearest") # Default colormap
    plt.title("Confusion Matrix")
    plt.colorbar(im)
    tick_marks = np.arange(len(label_order))
    plt.xticks(tick_marks, label_order, rotation=45)
    plt.yticks(tick_marks, label_order)
    # Annotate cells
    thresh = conf.max() / 2.0 if conf.max() > 0 else 0
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            plt.text(j, i, format(conf[i, j], "d"),
                     ha="center", va="center",
                     color="white" if conf[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Metrics table (per-class and global)
    metrics_rows = []
    for j, label in enumerate(label_order):
        # Compute binary confusion components for one-vs-rest
        y_true_bin_col = y_true_bin[:, j]
        y_pred_bin_col = (y_pred == label).astype(int)
        TP = int(((y_pred_bin_col == 1) & (y_true_bin_col == 1)).sum())
        TN = int(((y_pred_bin_col == 0) & (y_true_bin_col == 0)).sum())
        FP = int(((y_pred_bin_col == 1) & (y_true_bin_col == 0)).sum())
        FN = int(((y_pred_bin_col == 0) & (y_true_bin_col == 1)).sum())

        precision = precision_score(y_true_bin_col, y_pred_bin_col, 
                                    zero_division=0)
        recall = recall_score(y_true_bin_col, y_pred_bin_col, 
                              zero_division=0)
        f1 = f1_score(y_true_bin_col, y_pred_bin_col, zero_division=0)
        # Accuracy for one-vs-rest
        accuracy = (TP + TN) / max(1, (TP + TN + FP + FN))

        roc_auc_val = per_class_auc[label]

        metrics_rows.append({
            "label": label,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": (None if np.isnan(roc_auc_val) else float(roc_auc_val)),
            "n_true_samples": int(y_true_bin_col.sum()),
            "n_pred_samples": int((y_pred == label).sum()),
        })

    # Global metrics
    global_accuracy = accuracy_score(y_true, y_pred)
    # Macro-precision/recall/f1 using sklearn
    global_precision_macro = precision_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0
    )
    global_recall_macro = recall_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0
    )
    global_f1_macro = f1_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0
    )

    # Add a global row
    metrics_rows.append({
        "label": "GLOBAL",
        "TP": None,
        "TN": None,
        "FP": None,
        "FN": None,
        "accuracy": float(global_accuracy),
        "precision": float(global_precision_macro),
        "recall": float(global_recall_macro),
        "f1": float(global_f1_macro),
        "roc_auc": None,
        "n_true_samples": int(len(y_true)),
        "n_pred_samples": None,
    })

    metrics_df = pd.DataFrame(metrics_rows)

    # Save metrics to CSV and JSON
    csv_path = output_dir / "metrics_table.csv"
    json_path = output_dir / "metrics_table.json"
    metrics_df.to_csv(csv_path, index=False)

    # For JSON, convert DataFrame to list of dicts with JSON serializable values
    metrics_json_ready = []
    for r in metrics_rows:
        r_copy = dict(r)
        # Convert numpy types to native python
        for k, v in r_copy.items():
            if isinstance(v, (np.generic,)):
                r_copy[k] = v.item()
            # Convert NaN to None
            if isinstance(v, float) and (np.isnan(v)):
                r_copy[k] = None
        metrics_json_ready.append(r_copy)

    with open(json_path, "w") as fjson:
        json.dump(metrics_json_ready, fjson, indent=2)

    # Create PDF report with the three images and a metrics page
    pdf_path = output_dir / "evaluation_report.pdf"
    with PdfPages(pdf_path) as pdf:
        # Title / Summary page
        fig = plt.figure(figsize=(8.27, 11.69)) # A4 portrait in inches
        plt.axis("off")
        title_lines = []
        if pdf_title:
            title_lines.append(pdf_title)
        title_lines.append("Model evaluation report")
        title_lines.append(f"Model path: {model_path}")
        title_lines.append(f"Number of samples: {len(y_true)}")
        title_lines.append("")
        title_lines.append(f"Labels (canonical order): {label_order}")
        title_lines.append(f"Macro AUC (mean of per-class AUCs): {macro_auc:.4f}" if not np.isnan(macro_auc) else "Macro AUC: N/A")
        title_lines.append(f"Micro AUC: {micro_auc:.4f}" if not np.isnan(micro_auc) else "Micro AUC: N/A")
        txt = "\n".join(title_lines)
        plt.text(0.5, 0.5, txt, ha="center", va="center", wrap=True, 
                 fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Append per-class ROC image
        try:
            img = plt.imread(str(roc_per_class_path))
            fig = plt.figure(figsize=(8.27, 6))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()
        except Exception:
            pass

        # Append macro ROC
        try:
            img = plt.imread(str(roc_macro_path))
            fig = plt.figure(figsize=(8.27, 6))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()
        except Exception:
            pass

        # Append micro ROC
        try:
            img = plt.imread(str(roc_micro_path))
            fig = plt.figure(figsize=(8.27, 6))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()
        except Exception:
            pass

        # Append confusion matrix
        try:
            img = plt.imread(str(cm_path))
            fig = plt.figure(figsize=(8.27, 6))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close()
        except Exception:
            pass

        # Append a metrics table page (raw text)
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        # Convert metrics_df to text nicely
        tbl_str = metrics_df.to_string(index=False)
        plt.text(0.01, 0.99, "Metrics table (CSV/JSON also saved):", 
                 ha="left", va="top", fontsize=10)
        plt.text(0.01, 0.95, tbl_str, ha="left", va="top", fontsize=8, 
                 family="monospace")
        pdf.savefig(fig)
        plt.close()

    # Return paths and a summary
    result = {
        "roc_per_class_png": str(roc_per_class_path),
        "roc_macro_png": str(roc_macro_path),
        "roc_micro_png": str(roc_micro_path),
        "confusion_matrix_png": str(cm_path),
        "metrics_csv": str(csv_path),
        "metrics_json": str(json_path),
        "pdf_report": str(pdf_path),
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "label_order": label_order,
    }
    return result


def full_evaluation_manager(
    model_path: Path,
    data_table_path: Path,
    output_folder: Path,
    logger: structlog.BoundLogger,
    target_variable: Optional[str]='Diagnosis',
    all_labels: Optional[List[str]] = None,
    pdf_title: Optional[str] = None,
):
    # Load data
    df = pd.read_csv(data_table_path, sep=',', index_col=0)

    X = df.T.drop(columns=[target_variable])
    y = df.T[target_variable]
    
    # Run full evaluation
    eval_out = full_evaluation(
        model_path=model_path,
        X=X,
        y_true=y,
        output_dir=output_folder,
        logger=logger,
        all_labels=all_labels,
        pdf_title=pdf_title
    )
    return None


def generate_roc_curves(y_true_int, y_pred_proba, class_labels, out_folder, out_name):
    """
    Generates one ROC curve per class.
    """

    n_classes = len(class_labels)

    # Convert y_true to one-hot
    y_true_bin = np.zeros((len(y_true_int), n_classes))
    for i, val in enumerate(y_true_int):
        y_true_bin[i, val] = 1

    for i, cls in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle='--')

        ax.set_title(f"ROC Curve — {cls}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

        roc_path = out_folder / f"{out_name}_ROC_{cls}.png"
        fig.tight_layout()
        fig.savefig(roc_path)
        plt.close(fig)


def get_single_feature_auc(
    X:pd.DataFrame,
    y:pd.Series,
    output_folder: Path,
    feature_name:str,
    model,
    rand_seed:int,
    max_auc_score:float=0.0
) -> float:
    """
    Trains a model using a single feature and returns the ROC-AUC 
    score.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Series of target labels.
        feature_name (str): The name of the feature to use.
        model: The machine learning model to train.
        rand_seed (int): Random seed for repeatability.
        max_auc_score (float): Maximum AUC score observed so far.
                               Used to skip saving files if the new
                               score is not better.

    Returns:
        float: The ROC-AUC score.
    """
    # Get the single feature from X
    X_single = X[[feature_name]]
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_single,
        y,
        test_size=0.3,
        random_state=rand_seed,
        stratify=y
    )
    
    # Handle multi-class classification for AUC
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    curr_model = model.fit(X_train, y_train)
    y_pred_proba = curr_model.predict_proba(X_test)

    # Calculate AUC score
    if y_test_bin.shape[1] == 1:
        auc_score = roc_auc_score(y_test_bin, y_pred_proba[:, 1])
    else:
        auc_score = roc_auc_score(y_test_bin, y_pred_proba, 
                                  multi_class='ovr')

    # If the AUC score is better than max_auc_score, 
    # delete previous files and save new ones
    if auc_score > max_auc_score:
        # Delete previous files
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))
        # Save train and test sets
        feat_name = feature_name.replace(' ', '_')
        X_train.to_csv(f'{output_folder}/{feat_name}_X_train.csv')
        X_test.to_csv(f'{output_folder}/{feat_name}_X_test.csv')
        y_train.to_csv(f'{output_folder}/{feat_name}_y_train.csv')
        y_test.to_csv(f'{output_folder}/{feat_name}_y_test.csv')
        # Save the model
        pickle_model(curr_model,
                    f'{output_folder}/{feat_name}_model.pickle')

    return auc_score


def load_pickled_model(model_path: Path):
    # Load trained model
    with open(model_path, "rb") as f:
        model_file = pickle.load(f)
    return model_file


def merge_with_metadata(
    feature_table: pd.DataFrame,
    metadata_file: Path,
    output_folder: Path,
    table_name: str,
    logger: structlog.BoundLogger
) -> pd.DataFrame:
    """
    Add metadata from a file to the feature table and save it.
    """
    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define output file name
    out_csv_transposed = os.path.join(output_folder, 
                                      f'{table_name}_transposed.csv')
    out_csv = os.path.join(output_folder, f'{table_name}.csv')

    logger.info('Opening metadata file.', metadata_file=metadata_file)

    # Open metadata file
    metadata_df = pd.read_csv(metadata_file, sep=',',
                              index_col=0)
    
    metadata_df = metadata_df.T

    # Modify gender to 1-0
    metadata_df.loc['Gender'] = \
        metadata_df.loc['Gender'].replace({'male': 0, 'female': 1})

    # Transpose tables
    feature_table = feature_table.T
    metadata_df = metadata_df.T

    logger.info('Attempting merge with feature table.',
                metadata_file=metadata_file)

    # Ensure both dataframes have the same index and are aligned
    if not feature_table.index.equals(metadata_df.index):
        logger.warning("DataFrames indices do not match. Merging...",
                       metadata_file=metadata_file)
        # This is a robust way to merge if indices don't match
        merged_df = pd.merge(metadata_df, feature_table,
                             left_index=True, right_index=True, 
                             how='inner')
    else:
        merged_df = pd.concat([metadata_df, feature_table], axis=1)

    logger.info('Merge completed. Saving files...',
                metadata_file=metadata_file)
    
    # Save df as csv
    merged_df.to_csv(out_csv, sep=',')
    merged_df.T.to_csv(out_csv_transposed, sep=',')
    return merged_df


def normalize_by_gv(
    df_in: pd.DataFrame,
    logger: structlog.BoundLogger
) -> pd.DataFrame:
    """
    Normalizes a pandas DataFrame by number of GVs per sample.
    """
    logger.info('Separating the different kinds of feature.')

    # Get df for the different kinds of feature
    bins_df = df_in.loc[
        df_in.index.to_series().str.startswith('bin_gvs')
    ]
    genes_df = df_in.loc[
        df_in.index.to_series().str.startswith('gene_gvs')
    ]
    dn_ds_df = df_in.loc[
        df_in.index.to_series().str.startswith('dn_ds')
    ]
    fl_df = df_in.loc[
        df_in.index.to_series().str.startswith('fragment')
    ]
    cnv_df = df_in.loc[
        df_in.index.to_series().str.startswith('cnv_length')
    ]

    logger.info('Dividing by number of GVs.')

    # Get the sum of gvs per column (from bins_df)
    gv_col_sum = (bins_df.sum())
    # Divide by total GVs
    bins_df = bins_df / gv_col_sum
    genes_df = genes_df / gv_col_sum

    logger.info('Re-joining separated dfs.')

    # Join dataframes into df
    df_out = pd.concat([dn_ds_df, fl_df, cnv_df, genes_df, bins_df], 
                        ignore_index=False)
    # Return df_out
    return df_out


def normalize_feature_table(
    input_file: Path,
    output_folder: Path,
    logger: structlog.BoundLogger,
    log_transform: bool = False,
    robust_norm: bool = False,
    rand_seed: int|None = None,
    parameters_file: Optional[Path] = None
) -> List[pd.DataFrame]:
    """
    Normalize a feature table and create heatmaps.
    
    Args:
        input_file (Path): Path to the feature table CSV file.
        output_folder (Path): Output directory for heatmaps and 
                              normalized tables.
        logger: Logger instance.
        log_transform (bool): Defines if logarithmic transform is
                              performed on the data.
        robust_norm (bool): Defines if robust normalization is used 
                            instead of subtracting the minimum value 
                            and dividing by the average.
    """
    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define table name from input_file
    table_name = input_file.name.rsplit('.')[0]

    # Parameter storage
    params_out = output_folder / "normalization_parameters.json"

    if parameters_file:
        logger.info("Loading normalization parameters.", file=str(parameters_file))
        with open(parameters_file, "r") as f:
            norm_params = json.load(f)
    else:
        norm_params = {}

    logger.info('Reading feature table.', table_path=input_file)

    # Read feature table
    df = pd.read_csv(input_file, sep=',')
    df.set_index(df.columns[0], inplace=True)

    # Define output file names
    out_raw = os.path.join(output_folder, f'{table_name}_RAW.csv')
    out_csv = os.path.join(output_folder, 
                           f'{table_name}_NORMALIZED.csv')

    if not parameters_file:
        df = cleanup_empty_rows_cols(df)

    # Copy raw df before numerical modifications
    raw_df = df.copy()

    if parameters_file:
        training_features = norm_params.get("feature_index_order")
        if training_features is None:
            raise ValueError(
                "Normalization parameters file does not contain 'feature_index_order'. "
                "Re-train normalization to include it."
            )

        # Add missing features (fill with zeros)
        for feat in training_features:
            if feat not in df.index:
                df.loc[feat] = 0

        # Remove extra features
        df = df.loc[df.index.intersection(training_features)]

        # Reorder to match training feature order exactly
        df = df.loc[training_features]

    df = normalize_by_gv(df, logger)

    if log_transform:
        logger.info('Perform log transformation.', 
                    table_path=input_file)
        # Perform log transformation
        df = df.apply(np.log1p)

    # Perform normalization
    if robust_norm:
        logger.info('Performing robust normalization.',
                    table_path=input_file)
        df, scaler_params = robust_normalize(
            df,
            loaded_params=norm_params.get("robust_scaler") if parameters_file else None
        )
        if not parameters_file:
            norm_params["robust_scaler"] = scaler_params
    else:
        logger.info('Performing normalization.',
                    table_path=input_file)
        if parameters_file:
            # Apply saved max-per-row
            df_max_per_row = pd.Series(norm_params["max_per_row"])
        else:
            df_max_per_row = df.max(axis=1)
            norm_params["max_per_row"] = df_max_per_row.to_dict()

        df = df.div(df_max_per_row, axis=0)

    if not parameters_file:
        df = cleanup_empty_rows_cols(df)

    logger.info('Removing minimums to avoid negative values.',
                table_path=input_file)

    # Remove minimums per row to avoid negative values
    if parameters_file:
        df_min_per_row = pd.Series(norm_params["min_per_row"])
    else:
        df_min_per_row = df.min(axis=1)
        norm_params["min_per_row"] = df_min_per_row.to_dict()

    df = df.sub(df_min_per_row, axis=0)

    if not parameters_file:
        df = cleanup_empty_rows_cols(df)

    # Save feature index order if not loaded from file
    if not parameters_file:
        norm_params["feature_index_order"] = df.index.tolist()

    logger.info('Creating heatmaps.', table_path=input_file)
    # Get sample of the dataframe
    df_sample = sample_df(df, rand_seed, n=10)
    # Change feature names for better visualization
    df_sample.index = df_sample.index.map(process_feature_names)
    # Create the sample heatmap
    create_heatmap(df_sample, table_name+'_sample', True, 
                   output_folder, logger=logger)

    logger.info('Modifying feature names and saving dataframes.',
                table_path=input_file)

    # Modify feature names
    df.index = df.index.map(process_feature_names)
    # Save df as csv
    df.to_csv(out_csv, sep=',')

    # Save raw_df too
    raw_df.index = raw_df.index.map(process_feature_names)
    raw_df.to_csv(out_raw, sep=',')
    
    # Save normalization parameters if not loaded from file
    if not parameters_file:
        with open(params_out, "w") as f:
            json.dump(norm_params, f, indent=2)
        logger.info("Saved normalization parameters.", file=str(params_out))

    # Return raw df and df
    return raw_df, df


def per_feature_analysis(
    table_name: str,
    output_folder: Path,
    target_var: str,
    logger: structlog.BoundLogger,
    top_n: int=20,
    rand_seed: int=None
) -> None:
    """Performs per-feature analysis on raw or normalized data."""
    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)
    # Define base name for outputs
    bname = str(table_name).rsplit('/')[-1].rsplit('.')[0]

    # Read feature table
    df = pd.read_csv(table_name, index_col=0)
    # Transpose df to have samples as rows and features as columns
    df = df.T
    
    model = RandomForestClassifier(random_state=rand_seed)

    logger.info("Analyzing feature table.", table_name=table_name)
    
    # Process features
    auc_results = analyze_features_and_rank_by_auc(
        df, output_folder, target_var, model, logger, rand_seed
    )

    logger.info("Feature analysis completed. Saving results.",
                table_name=table_name,
                feature_count=len(auc_results))
    
    # Define output file name
    file_name = f"{bname}_feature_singles.csv"
    file_out = output_folder / file_name

    # Save output
    auc_results.to_csv(file_out)

    if top_n >= 2:
        # Define top features from auc_results
        top_features = auc_results.head(top_n).index.tolist()

        logger.debug(f'Top {top_n} features obtained.',
                     top_features=top_features)

        logger.info(f'Analyzing top {top_n} features in pairs.',
                    table_name=table_name,
                    top_n=top_n)

        # Define pairs output name
        output_folder_pairs = output_folder / "pairs_auc_out"
        output_pairs = output_folder_pairs / \
            f"{bname}_feature_pairs.csv"

        # Run the analysis
        top_feature_pairs_ranked = analyze_feature_pairs(
            df,
            output_folder_pairs,
            top_features,
            target_var,
            logger,
            rand_seed
        )

        # Save the resulting dataframe
        top_feature_pairs_ranked.to_csv(output_pairs)
        
        if top_n >= 3:
            logger.info(f'Analyzing top {top_n} features in trios.',
                    table_name=table_name, top_n=top_n)
            # Define trios output name
            output_folder_trios = output_folder / "trios_auc_out"
            output_trios = output_folder_trios / \
                f"{bname}_feature_trios.csv"

            # Run the analysis for trios
            top_feature_trios_ranked = analyze_feature_trios(
                df,
                output_folder_trios,
                top_features,
                target_var,
                logger,
                rand_seed
            )

            # Save the resulting dataframe
            top_feature_trios_ranked.to_csv(output_trios)

    return None


def pickle_load(model_path: Path):
    """Loads a pickled model from the specified input path."""
    with open(model_path, "rb") as f:
        model_file = pickle.load(f)
    return model_file


def pickle_model(model, output_path: Path) -> None:
    """Pickles a model to the specified output path."""
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    return None


def plot_feature_roc(
    df: pd.DataFrame, 
    target_var: str, 
    output_folder: Path,
    feature_names: List[str],
    fitted_model: Any,
    logger: structlog.BoundLogger,
    random_state: int|None = None
) -> None:
    """
    Generates and displays a multi-class One-vs-Rest ROC curve plot 
    for one or more features using the provided model, evaluated on a 
    held-out test set.

    Args:
        df (pd.DataFrame): The full DataFrame.
        target_var (str): Name of the target variable column.
        output_folder (Path): Directory to store the final models.
        feature_names (List[str]): One or more feature names.
        fitted_model: The trained RF model for these features (fitted on full data).
        logger: Logger instance.
        random_state (int|None): Seed for reproducibility.
    """

    logger.info(f"Generating ROC curve for features: {feature_names} ")
    
    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define X and y
    X = df.drop(columns=[target_var])
    y = df[target_var]

    # Use LabelBinarizer for easy One-vs-Rest handling
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y)
    classes = label_binarizer.classes_
    n_classes = len(classes)
    
    # Prepare Data and Split
    X_feat = X[feature_names]
    
    # Split the data
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        X_feat, y,
        test_size=0.33, 
        stratify=y, 
        random_state=random_state
    )
    
    # Binarize the test labels
    y_test_bin = label_binarizer.transform(y_test_orig)

    # Predict probabilities using the provided fitted model
    y_proba = fitted_model.predict_proba(X_test)

    # Macro OVR AUC
    roc_auc_ovr_macro = roc_auc_score(
        y_test_orig,
        y_proba,
        multi_class="ovr",
        average="macro"
    )

    # Micro-averaged ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    # Macro-averaged ROC: average interpolated TPR across classes
    fpr = dict()
    tpr = dict()
    roc_auc_class = dict()

    # Collect all FPRs to build a common grid for macro
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_test_bin[:, i], y_proba[:, i])[0]
        for i in range(n_classes)
    ]))

    mean_tpr = np.zeros_like(all_fpr)

    # Compute per-class curves and accumulate for macro
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_class[i] = auc(fpr[i], tpr[i])
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    auc_macro_curve = auc(all_fpr, mean_tpr)

    # Per-class ROC curves
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(
            fpr[i], 
            tpr[i], 
            label=f'Class {classes[i]} (AUC = {roc_auc_class[i]:.2f})'
        )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Per-Class ROC Curves for {feature_names}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_folder / f'roc_per_class_{"_".join(feature_names)}.png')
    plt.close()

    # Micro-average ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (AUC = {auc_micro:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Micro-Average ROC Curve for {feature_names}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_folder / f'roc_micro_{"_".join(feature_names)}.png')
    plt.close()

    # Macro-average ROC
    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {auc_macro_curve:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Macro-Average ROC Curve for {feature_names}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(output_folder / f'roc_macro_{"_".join(feature_names)}.png')
    plt.close()

    logger.info(
        f"Generated ROC curves for features: {feature_names} "
        f"(Macro AUC: {roc_auc_ovr_macro:.3f})"
    )
    
    return None


def process_feature_names(row_name:str) -> str:
    """Processes a feature row name and formats it."""
    ret = row_name
    if row_name.startswith('bin_gvs'):
        region = row_name.rsplit('_')[-1]
        ret = f'GVs in region {region}'
    elif row_name.startswith('gene_gvs'):
        gene_name = row_name.rsplit('_')[-1]
        ret = f'GVs in gene {gene_name}'
    elif row_name.startswith('fragment'):
        feat = row_name.rsplit('_')[-1]
        if feat == 'mean':
            ret = 'Mean fragment length'
        elif feat == 'median':
            ret = 'Median fragment length'
        elif feat == 'std':
            ret = 'Fragment length standard deviation'
        elif feat == 'min':
            ret = 'Minimum fragment length'
        elif feat == 'max':
            ret = 'Maximum fragment length'
        elif feat == 'count':
            ret = 'Fragment count'
        else:
            er = f'WARNING: row_name {row_name} not processed properly.'
            print(er)
    elif row_name.startswith('cnv_length'):
        chr_n = row_name.rsplit('_')[-1]
        gain_loss = row_name.rsplit('_')[-2]
        if gain_loss == 'gain':
            ret = f'Nucleotides added by CNVs in {chr_n}'
        elif gain_loss == 'loss':
            ret = f'Nucleotides lost by CNVs in {chr_n}'
        else:
            er = f'WARNING: row_name {row_name} not processed properly.'
            print(er)
    elif row_name.startswith('dn_ds') or row_name.startswith('gene_dn_ds'):
        gene_name = row_name.rsplit('_')[-1]
        ret = f'dN/dS proportion in gene {gene_name}'
    else:
        print(f'row_name {row_name} not processed.')
    return ret


def robust_normalize(
    df: pd.DataFrame,
    loaded_params: Optional[dict] = None
):
    """
    Performs robust normalization on a dataframe.
    If loaded_params is given, use them.
    """
    q1 = df.quantile(0.25, axis=1)
    q3 = df.quantile(0.75, axis=1)
    rows_to_keep = (q3 != 0) | (q1 != 0)
    df_filtered = df[rows_to_keep]

    df_transposed = df_filtered.T

    scaler = RobustScaler()

    if loaded_params:
        scaler.center_ = np.array(loaded_params["center_"])
        scaler.scale_ = np.array(loaded_params["scale_"])
        scaled_data = scaler.transform(df_transposed)
    else:
        scaled_data = scaler.fit_transform(df_transposed)
        loaded_params = {
            "center_": scaler.center_.tolist(),
            "scale_": scaler.scale_.tolist()
        }

    df_row_scaled = pd.DataFrame(scaled_data.T,
                                 columns=df_filtered.columns,
                                 index=df_filtered.index)

    return df_row_scaled, loaded_params


def run_model_validation_and_test(
    model_path: Path,
    data_table_path: Path,
    out_folder: Path,
    target_var: str,
    logger: structlog.BoundLogger,
    out_name: str='',
    feature_names: List[str]=[]
):
    """
    Runs a model on a data table and creates performance metrics.
    Includes the creation of a confusion matrix figure.
    """
    # Make sure output folder exists
    out_folder.mkdir(parents=True, exist_ok=True)

    logger.info('Loading model.', model_path=model_path)

    # Load model
    model = pickle_load(model_path)
    # Define basename from model_path
    bname = model_path.name.rsplit('.')[0]

    logger.info('Reading data table.', data_table=data_table_path)

    # Load data table
    df = pd.read_csv(data_table_path, sep=',', index_col=0)
    # Define X and y values
    X = df.T.drop(columns=[target_var])
    y = df.T[target_var]
    # Binarize y values
    ALL_POSSIBLE_CLASSES = ['Healthy', 'CRC', 'BRC'] 
    lb = LabelBinarizer()
    lb.fit(ALL_POSSIBLE_CLASSES)
    y_true_bin = lb.transform(y)

    print('Model params:', model.get_params())
    print('Feature importances:', model.feature_importances_)

    # Select only the features used in the model
    if feature_names:
        model_features = feature_names
    else:
        model_features = model.feature_names_in_
    X = X[model_features]

    logger.info('Evaluating model on data table.',
                data_table=data_table_path,
                feature_count=len(model_features),
                sample_count=len(X))

    # Evaluate the model on the test set
    y_pred = model.predict_proba(X)

    # Define output confusion matrix path
    out_conf = out_folder / f'conf_matrix_{bname}{out_name}.png'
    # Transform one-hot encoded lists to labels
    y_true_labels = np.argmax(y_true_bin, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    logger.info('Creating confusion matrix.',
                data_table=data_table_path,
                model_path=model_path)
    
    # Create confusion matrix figures
    create_conf_matrix(y_true_labels, y_pred_labels, 
                       ALL_POSSIBLE_CLASSES, out_conf)
    
    logger.info('Calculating performance metrics.',
                data_table=data_table_path,
                model_path=model_path)
    
    # Calculate AUC score
    auc_score = roc_auc_score(y_true_bin, y_pred,
                              multi_class='ovr')
    
    # Create ROC curve figure
    create_auc_roc_curves(
        y_true_bin,
        y_pred,
        out_folder,
        bname,
        out_name,
        model,
        logger,
    )

    create_micro_avg_auc_roc_curve(
        y_true_bin,
        y_pred,
        out_folder,
        bname,
        out_name,
        logger
    )

    create_macro_avg_auc_roc_curve(
        y_true_bin,
        y_pred,
        out_folder,
        bname,
        out_name,
        logger
    )

    # Obtain accuracy, precision, recall, f1
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels,
                                average='weighted', zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels,
                          average='weighted', zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels,
                  average='weighted', zero_division=0)
    
    logger.info('Performance metrics calculated. Saving results.',
                data_table=data_table_path,
                model_path=model_path)
    
    # Save results to a text file
    out_txt = out_folder / f'performance_{bname}{out_name}.txt'
    with open(out_txt, 'w') as f:
        f.write(f'AUC: {auc_score}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1: {f1}\n')
    # Return the performance metrics into a dictionary
    results = {
        'AUC': auc_score,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    return results



def sample_df(
    df: pd.DataFrame,
    rand_seed: int|None,
    n: int=10
) -> pd.DataFrame:
    """
    Takes a sample of n rows from each feature type.
    """
    # Get df for the different kinds of feature
    bins_df = df.loc[df.index.to_series().str.startswith('bin_gvs')]
    genes_df = df.loc[df.index.to_series().str.startswith('gene_gvs')]
    dn_ds_df = df.loc[df.index.to_series().str.startswith('dn_ds')]
    fl_df = df.loc[df.index.to_series().str.startswith('fragment')]
    cnv_df = df.loc[df.index.to_series().str.startswith('cnv_length')]
    # Create a heatmap with samples from subsets of features
    l_df = [
        fl_df,
        cnv_df.sample(n=n, random_state=rand_seed),
        genes_df.sample(n=n, random_state=rand_seed),
        bins_df.sample(n=n, random_state=rand_seed)
    ]
    if not dn_ds_df.empty:
        l_df.append(dn_ds_df.sample(n=n, random_state=rand_seed))
    df_sample = pd.concat(l_df, ignore_index=False)

    return df_sample


def top_single_features_cvrf(
    df: pd.DataFrame,
    target_var: str,
    logger: structlog.BoundLogger,
    split_n: int = 5,
    random_state: int | None = None
) -> pd.DataFrame:
    """
    Performs cross-validated Random Forest (RF) classification on each
    feature and returns a DataFrame with metrics:
    - mean AUC
    - std AUC
    - mean FPR, FNR, TPR, TNR

    Multi-class AUC is computed with OVR macro averaging.
    """

    # Define X and y
    X = df.drop(columns=[target_var])
    y = df[target_var]

    logger.info(
        f"Starting single-feature CVRF on {X.shape[1]} features..."
    )

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    if n_classes < 2:
        logger.error("Target variable must have at least two unique classes.")
        return pd.DataFrame()

    # Cross-validation setup
    cv = StratifiedKFold(
        n_splits=split_n,
        shuffle=True,
        random_state=random_state
    )

    # Base RF model
    rf_base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )

    # Store results per feature
    results = {}

    # Display
    cont = 0
    n_col = len(X.columns)

    # Iterate through each feature
    for feature_name in X.columns:
        # Select one feature
        X_feature = X[[feature_name]].values

        fold_auc_scores = []
        fold_fpr = []
        fold_fnr = []
        fold_tpr = []
        fold_tnr = []

        # CV loop
        for fold, (train_index, test_index) in enumerate(
                cv.split(X_feature, y_encoded)):

            X_train = X_feature[train_index]
            X_test = X_feature[test_index]
            y_train = y_encoded[train_index]
            y_test = y_encoded[test_index]

            # Clone model to avoid leakage across folds
            rf_model = clone(rf_base_model)

            try:
                # Train
                rf_model.fit(X_train, y_train)

                # Predict probabilities (for AUC)
                y_proba = rf_model.predict_proba(X_test)

                # AUC
                auc = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class='ovr',
                    average='macro',
                    labels=np.arange(n_classes)
                )
                fold_auc_scores.append(auc)

                # Predict classes (for confusion matrix)
                y_pred = rf_model.predict(X_test)

                # Compute confusion matrix
                cm = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=np.arange(n_classes)
                )

                # For multi-class, compute averages over classes:
                # sensitivity, specificity, etc. per class, then mean
                fpr_list = []
                fnr_list = []
                tpr_list = []
                tnr_list = []

                for c in range(n_classes):
                    # One-vs-all counts
                    TP = cm[c, c]
                    FN = cm[c, :].sum() - TP
                    FP = cm[:, c].sum() - TP
                    TN = cm.sum() - (TP + FP + FN)

                    # Avoid division errors
                    # (required for stability, not a silent failure)
                    if TP + FN > 0:
                        tpr_list.append(TP / (TP + FN))
                        fnr_list.append(FN / (TP + FN))
                    if TN + FP > 0:
                        tnr_list.append(TN / (TN + FP))
                        fpr_list.append(FP / (TN + FP))

                # Fold-level mean metrics
                fold_fpr.append(np.mean(fpr_list))
                fold_fnr.append(np.mean(fnr_list))
                fold_tpr.append(np.mean(tpr_list))
                fold_tnr.append(np.mean(tnr_list))

            except ValueError as e:
                logger.warning(
                    f"Skipping fold {fold} for feature {feature_name} "
                    f"due to error: {e}"
                )

        # Display progress every 1000 features
        if cont == 0 or (cont + 1) % 1000 == 0:
            logger.debug(f"Progress: {cont + 1} / {n_col}")
        cont += 1

        # Aggregate metrics for this feature
        if fold_auc_scores:
            results[feature_name] = {
                "mean_auc": np.mean(fold_auc_scores),
                "std_auc": np.std(fold_auc_scores),
                "mean_fpr": np.mean(fold_fpr),
                "mean_fnr": np.mean(fold_fnr),
                "mean_tpr": np.mean(fold_tpr),
                "mean_tnr": np.mean(fold_tnr),
            }
        else:
            results[feature_name] = {
                "mean_auc": np.nan,
                "std_auc": np.nan,
                "mean_fpr": np.nan,
                "mean_fnr": np.nan,
                "mean_tpr": np.nan,
                "mean_tnr": np.nan,
            }

    # Convert to DataFrame
    df_results = pd.DataFrame.from_dict(results, orient="index")

    # Sort by mean AUC descending
    df_results = df_results.sort_values(by="mean_auc", ascending=False)

    logger.info("CVRF classification complete.")

    return df_results


def train_and_store_final_models(
    df: pd.DataFrame,
    target_var: str,
    output_folder: Path,
    top_feature_names: List[str],
    features_metrics: pd.DataFrame,
    logger: structlog.BoundLogger,
    random_state: int|None = None
) -> Dict:
    """
    Trains a final Random Forest model on the entire dataset for each
    of the top performing features and stores the model object.

    Args:
        df (pd.DataFrame): The full DataFrame.
        target_var (str): Name of the target variable column.
        output_folder (Path): Directory to store the final models.
        top_feature_names (List[str]): List of feature names to train
                                       final models for.
        features_metrics (pd.DataFrame): DataFrame of CVRF metrics per 
                                         feature.
        logger: Logger instance.
        random_state (int|None): Seed for reproducibility.

    Returns:
        A dictionary mapping feature name to a tuple: 
        (mean_cv_auc, fitted_model, model_path, feature_list,
         full_dataset_metrics_dict)
    """
    # Make sure output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define X and y
    X = df.drop(columns=[target_var])
    y = df[target_var]

    final_models = {}
    
    # Base model
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5, 
        class_weight='balanced', 
        random_state=random_state,
        n_jobs=-1
    )

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logger.info("Training and storing models for the " +
                f"top {len(top_feature_names)} features...")
    
    # Train a model on the full data for each top feature
    for feature_name in top_feature_names:
        X_feature = X[[feature_name]].values

        # Clone the base model
        rf_final_model = clone(rf_model)

        # Train the model
        rf_final_model.fit(X_feature, y_encoded)
        
        # Pickle the model
        feat_name = feature_name.replace(' ', '_')
        model_path = output_folder / f'{feat_name}_model.pickle'
        pickle_model(rf_final_model, model_path)

        # Retrieve previously computed CV AUC from features_metrics
        mean_cv_auc = features_metrics.loc[feature_name, 'mean_auc']

        # Compute performance metrics
        y_pred = rf_final_model.predict(X_feature)
        y_proba = rf_final_model.predict_proba(X_feature)

        # ROC-AUC (multiclass OvR)
        try:
            roc_auc_full = roc_auc_score(
                y_encoded, y_proba, multi_class='ovr'
            )
        except ValueError:
            logger.warning(
                f"Could not compute ROC-AUC for feature {feature_name}"
            )
            roc_auc_full = None

        accuracy_full = accuracy_score(y_encoded, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_encoded, y_pred)

        # Convert confusion matrix into a structured dict
        class_labels = le.classes_
        confusion_details = {}
        for i, cls in enumerate(class_labels):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            confusion_details[cls] = {
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "TN": int(TN)
            }

        metrics_full = {
            "roc_auc_full": roc_auc_full,
            "accuracy_full": accuracy_full,
            "confusion_per_class": confusion_details
        }

        # Store everything
        final_models[feature_name] = (
            mean_cv_auc,        # CV ROC-AUC
            rf_final_model,     # fitted model
            model_path,         # where it was saved
            [feature_name],     # list of features used
            metrics_full        # metrics computed on full dataset
        )
        
    return final_models


def train_and_store_final_models_for_groups(
    df: pd.DataFrame,
    target_var: str,
    output_folder: Path,
    top_group_ids: List[str],
    features_metrics: pd.DataFrame,
    logger: structlog.BoundLogger,
    random_state: int | None = None
) -> Dict:
    """
    Trains a final Random Forest model on the entire dataset for each
    selected feature group (single feature, pairs, trios, etc.) and
    stores the resulting model. Receives a list of Feature Group IDs 
    that match rows in the features_metrics table.

    Args:
        df (pd.DataFrame): The full DataFrame used to train models.
        target_var (str): Column name for the target variable.
        output_folder (Path): Directory to save final trained models.
        top_group_ids (List[str]): List of Feature Group IDs to train
                                   final models for.
        features_metrics (pd.DataFrame): DataFrame containing per-group
                                         CV metrics and the list of
                                         features (Feature 1..N). 
                                         Feature Group IDs is the index.
        logger: Logger instance.
        random_state (int|None): Seed for reproducibility.

    Returns:
        A dictionary mapping Feature Group ID → tuple:
            (
                mean_cv_auc,             # CV ROC-AUC from metrics table
                fitted_model,            # trained RandomForest model
                model_path,              # path where it was saved
                feature_list,            # actual list of feature names
                full_dataset_metrics     # dict: AUC, accuracy, confusion
            )
    """
    # Ensure folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Split predictors and target
    X = df.drop(columns=[target_var])
    y = df[target_var]

    final_models = {}

    # Base classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    logger.info("Training final models for "
                f"{len(top_group_ids)} feature groups...")

    # Identify feature columns in metrics table
    feature_cols = [c for c in features_metrics.columns
                    if c.startswith("Feature")]

    # Iterate through feature groups
    for group_id in top_group_ids:
        # Extract row for this group
        if group_id not in features_metrics.index:
            raise KeyError(
                f"Group ID '{group_id}' not found in features_metrics"
            )

        row = features_metrics.loc[group_id]

        # Extract feature list (drop NaN)
        feature_list = [
            str(row[c]) for c in feature_cols
            if pd.notna(row[c])
        ]

        # Extract CV AUC
        mean_cv_auc = row["AUC Mean"]

        # Subset data to the group's features
        X_group = X[feature_list].values

        # Train model
        rf_final_model = clone(rf_model)
        rf_final_model.fit(X_group, y_encoded)

        # Save model
        safe_group_name = group_id.replace(" ", "_")
        model_path = output_folder / f'{safe_group_name}_model.pickle'
        pickle_model(rf_final_model, model_path)

        # Evaluate on dataset
        y_pred = rf_final_model.predict(X_group)
        y_proba = rf_final_model.predict_proba(X_group)

        # ROC-AUC (OvR multiclass)
        try:
            roc_auc_full = roc_auc_score(
                y_encoded, y_proba, multi_class='ovr'
            )
        except ValueError:
            logger.warning(
                f"Could not compute ROC-AUC for group {group_id}"
            )
            roc_auc_full = None

        accuracy_full = accuracy_score(y_encoded, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_encoded, y_pred)

        # Per-class stats
        class_labels = le.classes_
        confusion_details = {}
        for i, cls in enumerate(class_labels):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)

            # Compute rates safely
            tpr = TP / (TP + FN) if (TP + FN) > 0 else None
            fnr = FN / (TP + FN) if (TP + FN) > 0 else None
            fpr = FP / (FP + TN) if (FP + TN) > 0 else None
            tnr = TN / (FP + TN) if (FP + TN) > 0 else None

            confusion_details[cls] = {
                "Mean TPR": tpr,
                "Mean FNR": fnr,
                "Mean FPR": fpr,
                "Mean TNR": tnr
            }

        metrics_full = {
            "roc_auc_full": roc_auc_full,
            "accuracy_full": accuracy_full,
            "confusion_per_class": confusion_details
        }

        # Store result
        final_models[group_id] = (
            mean_cv_auc,
            rf_final_model,
            model_path,
            feature_list,
            metrics_full
        )

    return final_models
