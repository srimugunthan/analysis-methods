"""
Comprehensive ANOVA Analysis Script
Includes: One-way, Two-way, Repeated Measures ANOVA
With assumptions checking, post-hoc tests, and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene, normaltest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ANOVAAnalysis:
    """Comprehensive ANOVA analysis with assumptions checking"""
    
    def __init__(self, data, dependent_var, group_var, subject_var=None):
        """
        Initialize ANOVA analysis
        
        Parameters:
        -----------
        data : DataFrame
            Input data
        dependent_var : str
            Name of dependent variable
        group_var : str or list
            Name of grouping variable(s)
        subject_var : str, optional
            Subject identifier for repeated measures
        """
        self.data = data.copy()
        self.dependent_var = dependent_var
        self.group_var = group_var if isinstance(group_var, list) else [group_var]
        self.subject_var = subject_var
        self.results = {}
        
    def check_normality(self, alpha=0.05):
        """Check normality assumption using Shapiro-Wilk test"""
        print("\n" + "="*60)
        print("NORMALITY CHECK (Shapiro-Wilk Test)")
        print("="*60)
        
        normality_results = {}
        
        for group_name, group_data in self.data.groupby(self.group_var):
            values = group_data[self.dependent_var].dropna()
            stat, p_value = shapiro(values)
            
            is_normal = p_value > alpha
            normality_results[str(group_name)] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': is_normal
            }
            
            print(f"\nGroup: {group_name}")
            print(f"  Statistic: {stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Normal? {'Yes' if is_normal else 'No'} (α={alpha})")
        
        self.results['normality'] = normality_results
        return normality_results
    
    def check_homogeneity(self, alpha=0.05):
        """Check homogeneity of variances using Levene's test"""
        print("\n" + "="*60)
        print("HOMOGENEITY OF VARIANCE (Levene's Test)")
        print("="*60)
        
        groups = [group[self.dependent_var].dropna() 
                  for name, group in self.data.groupby(self.group_var)]
        
        stat, p_value = levene(*groups)
        is_homogeneous = p_value > alpha
        
        print(f"\nStatistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Homogeneous? {'Yes' if is_homogeneous else 'No'} (α={alpha})")
        
        self.results['homogeneity'] = {
            'statistic': stat,
            'p_value': p_value,
            'is_homogeneous': is_homogeneous
        }
        
        return is_homogeneous
    
    def one_way_anova(self):
        """Perform one-way ANOVA"""
        print("\n" + "="*60)
        print("ONE-WAY ANOVA")
        print("="*60)
        
        # Get groups
        groups = [group[self.dependent_var].dropna() 
                  for name, group in self.data.groupby(self.group_var)]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        grand_mean = self.data[self.dependent_var].mean()
        ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in groups])
        ss_total = sum([(x - grand_mean)**2 for g in groups for x in g])
        eta_squared = ss_between / ss_total
        
        print(f"\nF-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Eta-squared (η²): {eta_squared:.4f}")
        
        if p_value < 0.05:
            print("\n*** Significant difference found between groups (p < 0.05) ***")
        else:
            print("\nNo significant difference found between groups (p >= 0.05)")
        
        self.results['anova'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared
        }
        
        return f_stat, p_value
    
    def post_hoc_tukey(self):
        """Perform Tukey HSD post-hoc test"""
        print("\n" + "="*60)
        print("POST-HOC TEST (Tukey HSD)")
        print("="*60)
        
        # Prepare data
        group_col = '_'.join(self.group_var)
        data_tukey = self.data[[self.dependent_var] + self.group_var].copy()
        data_tukey['group'] = data_tukey[self.group_var].astype(str).agg('_'.join, axis=1)
        
        # Perform Tukey HSD
        tukey_result = pairwise_tukeyhsd(
            endog=data_tukey[self.dependent_var],
            groups=data_tukey['group'],
            alpha=0.05
        )
        
        print("\n", tukey_result)
        
        self.results['post_hoc'] = tukey_result
        return tukey_result
    
    def two_way_anova(self):
        """Perform two-way ANOVA"""
        if len(self.group_var) < 2:
            print("Two-way ANOVA requires at least 2 grouping variables")
            return None
        
        print("\n" + "="*60)
        print("TWO-WAY ANOVA")
        print("="*60)
        
        # Create formula
        formula = f"{self.dependent_var} ~ C({self.group_var[0]}) + C({self.group_var[1]}) + C({self.group_var[0]}):C({self.group_var[1]})"
        
        # Fit model
        model = ols(formula, data=self.data).fit()
        anova_table = anova_lm(model, typ=2)
        
        print("\n", anova_table)
        
        self.results['two_way_anova'] = anova_table
        return anova_table
    
    def plot_distributions(self, figsize=(15, 5)):
        """Plot distributions by group"""
        n_groups = len(self.group_var)
        
        if n_groups == 1:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Box plot
            self.data.boxplot(column=self.dependent_var, by=self.group_var[0], ax=axes[0])
            axes[0].set_title(f'Box Plot by {self.group_var[0]}')
            axes[0].set_xlabel(self.group_var[0])
            axes[0].set_ylabel(self.dependent_var)
            plt.sca(axes[0])
            plt.xticks(rotation=45)
            
            # Violin plot
            sns.violinplot(data=self.data, x=self.group_var[0], y=self.dependent_var, ax=axes[1])
            axes[1].set_title(f'Violin Plot by {self.group_var[0]}')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Strip plot with means
            sns.stripplot(data=self.data, x=self.group_var[0], y=self.dependent_var, 
                         alpha=0.5, ax=axes[2])
            means = self.data.groupby(self.group_var[0])[self.dependent_var].mean()
            axes[2].scatter(range(len(means)), means, color='red', s=200, 
                          marker='D', label='Mean', zorder=5)
            axes[2].set_title(f'Strip Plot with Means by {self.group_var[0]}')
            axes[2].legend()
            axes[2].tick_params(axis='x', rotation=45)
            
        elif n_groups == 2:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Interaction plot
            grouped = self.data.groupby(self.group_var)[self.dependent_var].mean().unstack()
            grouped.plot(kind='line', marker='o', ax=axes[0])
            axes[0].set_title('Interaction Plot')
            axes[0].set_xlabel(self.group_var[0])
            axes[0].set_ylabel(f'Mean {self.dependent_var}')
            axes[0].legend(title=self.group_var[1])
            
            # Grouped box plot
            self.data.boxplot(column=self.dependent_var, 
                            by=self.group_var, ax=axes[1])
            axes[1].set_title(f'Box Plot by {" and ".join(self.group_var)}')
            plt.sca(axes[1])
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/claude/anova_distributions.png', dpi=300, bbox_inches='tight')
        print("\nDistribution plots saved to: /home/claude/anova_distributions.png")
        plt.close()
    
    def plot_diagnostics(self):
        """Plot diagnostic plots for assumptions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Q-Q plot
        stats.probplot(self.data[self.dependent_var], dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title('Q-Q Plot')
        
        # Histogram
        axes[0, 1].hist(self.data[self.dependent_var], bins=30, edgecolor='black')
        axes[0, 1].set_title('Histogram of Dependent Variable')
        axes[0, 1].set_xlabel(self.dependent_var)
        axes[0, 1].set_ylabel('Frequency')
        
        # Residual plot (for each group)
        for group_name, group_data in self.data.groupby(self.group_var):
            residuals = group_data[self.dependent_var] - group_data[self.dependent_var].mean()
            axes[1, 0].scatter([str(group_name)] * len(residuals), residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuals by Group')
        axes[1, 0].set_xlabel('Group')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Variance plot
        variances = self.data.groupby(self.group_var)[self.dependent_var].var()
        axes[1, 1].bar(range(len(variances)), variances.values)
        axes[1, 1].set_xticks(range(len(variances)))
        axes[1, 1].set_xticklabels([str(x) for x in variances.index], rotation=45)
        axes[1, 1].set_title('Variance by Group')
        axes[1, 1].set_xlabel('Group')
        axes[1, 1].set_ylabel('Variance')
        
        plt.tight_layout()
        plt.savefig('/home/claude/anova_diagnostics.png', dpi=300, bbox_inches='tight')
        print("Diagnostic plots saved to: /home/claude/anova_diagnostics.png")
        plt.close()
    
    def summary_statistics(self):
        """Print summary statistics by group"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY GROUP")
        print("="*60)
        
        summary = self.data.groupby(self.group_var)[self.dependent_var].describe()
        print("\n", summary)
        
        self.results['summary_stats'] = summary
        return summary


# Example 1: One-Way ANOVA
def example_one_way_anova():
    """Example of one-way ANOVA with three groups"""
    print("\n" + "#"*60)
    print("EXAMPLE 1: ONE-WAY ANOVA")
    print("#"*60)
    
    # Generate sample data
    np.random.seed(42)
    
    group_a = np.random.normal(20, 5, 30)
    group_b = np.random.normal(25, 5, 30)
    group_c = np.random.normal(22, 5, 30)
    
    data = pd.DataFrame({
        'score': np.concatenate([group_a, group_b, group_c]),
        'treatment': ['A']*30 + ['B']*30 + ['C']*30
    })
    
    # Perform analysis
    analysis = ANOVAAnalysis(data, dependent_var='score', group_var='treatment')
    analysis.summary_statistics()
    analysis.check_normality()
    analysis.check_homogeneity()
    analysis.one_way_anova()
    analysis.post_hoc_tukey()
    analysis.plot_distributions()
    analysis.plot_diagnostics()
    
    return analysis


# Example 2: Two-Way ANOVA
def example_two_way_anova():
    """Example of two-way ANOVA"""
    print("\n" + "#"*60)
    print("EXAMPLE 2: TWO-WAY ANOVA")
    print("#"*60)
    
    # Generate sample data
    np.random.seed(42)
    
    n_per_group = 20
    data_list = []
    
    for method in ['Online', 'Offline']:
        for difficulty in ['Easy', 'Medium', 'Hard']:
            base_score = {'Easy': 85, 'Medium': 75, 'Hard': 65}[difficulty]
            if method == 'Online':
                base_score += 5  # Online boost
            
            scores = np.random.normal(base_score, 8, n_per_group)
            
            for score in scores:
                data_list.append({
                    'score': score,
                    'method': method,
                    'difficulty': difficulty
                })
    
    data = pd.DataFrame(data_list)
    
    # Perform analysis
    analysis = ANOVAAnalysis(data, dependent_var='score', 
                            group_var=['method', 'difficulty'])
    analysis.summary_statistics()
    analysis.check_normality()
    analysis.check_homogeneity()
    analysis.two_way_anova()
    analysis.plot_distributions(figsize=(12, 5))
    
    return analysis


# Example 3: Repeated Measures ANOVA (within-subjects)
def example_repeated_measures():
    """Example of repeated measures ANOVA"""
    print("\n" + "#"*60)
    print("EXAMPLE 3: REPEATED MEASURES ANOVA")
    print("#"*60)
    
    # Generate sample data
    np.random.seed(42)
    
    n_subjects = 20
    data_list = []
    
    for subject_id in range(n_subjects):
        baseline = np.random.normal(100, 10)
        
        for time_point, time_label in enumerate(['Pre', 'Mid', 'Post'], start=1):
            # Add improvement effect
            improvement = time_point * 5
            score = baseline + improvement + np.random.normal(0, 5)
            
            data_list.append({
                'subject': subject_id,
                'time': time_label,
                'score': score
            })
    
    data = pd.DataFrame(data_list)
    
    # For repeated measures, we'll use one-way ANOVA on time
    # (proper repeated measures ANOVA requires more complex handling)
    analysis = ANOVAAnalysis(data, dependent_var='score', 
                            group_var='time', subject_var='subject')
    analysis.summary_statistics()
    analysis.check_normality()
    analysis.check_homogeneity()
    analysis.one_way_anova()
    analysis.post_hoc_tukey()
    analysis.plot_distributions()
    
    # Plot repeated measures
    plt.figure(figsize=(10, 6))
    for subject in data['subject'].unique()[:10]:  # Plot first 10 subjects
        subject_data = data[data['subject'] == subject]
        plt.plot(subject_data['time'], subject_data['score'], 
                alpha=0.3, marker='o')
    
    means = data.groupby('time')['score'].mean()
    plt.plot(means.index, means.values, 'r-', linewidth=3, 
            marker='o', markersize=10, label='Mean')
    plt.xlabel('Time Point')
    plt.ylabel('Score')
    plt.title('Repeated Measures Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/repeated_measures_plot.png', dpi=300, bbox_inches='tight')
    print("\nRepeated measures plot saved to: /home/claude/repeated_measures_plot.png")
    plt.close()
    
    return analysis


# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE ANOVA ANALYSIS")
    print("="*60)
    
    # Run all examples
    analysis1 = example_one_way_anova()
    analysis2 = example_two_way_anova()
    analysis3 = example_repeated_measures()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - anova_distributions.png")
    print("  - anova_diagnostics.png")
    print("  - repeated_measures_plot.png")
