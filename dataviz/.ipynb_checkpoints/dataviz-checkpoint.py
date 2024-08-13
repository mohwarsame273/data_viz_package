import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataViz:
    def __init__(self, dataframe):
        """
        Initialize with a pandas DataFrame.
        """
        self.dataframe = dataframe
        self.background_colour = '#DDDFDF'
        sns.set(rc={'axes.facecolor': self.background_colour, 'figure.facecolor': self.background_colour})

    def bar_plots(self, variables, target_variable, title, commentary, axs, highlight_positions=None):
        """
        Create bar plots for a list of variables against a target variable.
        """
        data = {}
        for variable in variables:
            data[variable] = (
                self.dataframe.query(f'{target_variable} == 1')
                .groupby(variable)
                .agg(churn_value=('Churn', 'mean'))  # Adjusted for churn rate
                .reset_index()
            )

        # Plot each variable
        for i, variable in enumerate(variables):
            sns.barplot(data=data[variable], x=variable, y='churn_value', color='#0F69FF', ax=axs[i],
                        alpha=0.8, lw=2, zorder=3, ec='black')

            axs[i].grid(which='major', axis='y', dashes=(1, 3), zorder=4, color='gray', ls=':')
            axs[i].grid(axis='x', visible=False)
            axs[i].set_ylabel('')
            axs[i].tick_params(labelsize=6, pad=0)
            axs[i].set_title(variable, weight='bold', fontsize=8)
            axs[i].xaxis.get_label().set_fontsize(8)
            axs[i].set_xlabel('')
            axs[i].yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}%'))

            for direction in ['top', 'right', 'left']:
                axs[i].spines[direction].set_visible(False)

        # Title & Commentary
        axs[0].text(-1, 330, title, fontweight='bold', fontsize=14, fontfamily='monospace')
        axs[0].text(-0.75, 285, commentary, fontweight='light', fontsize=10, fontfamily='monospace', va='center')

        if highlight_positions:
            bbox_props = dict(boxstyle="circle, pad=0.3", fc='#11D3CD', ec='#11D3CD')
            for pos in highlight_positions:
                axs[pos[0]].text(pos[1], pos[2], pos[3], ha='center', va='center', size=8, bbox=bbox_props)

    def upset_plot(self, indicator_columns, title, commentary, highlight_positions=None):
        """
        Create an UpSet plot to show high-risk cohorts.
        """
        upset_raw = pd.get_dummies(self.dataframe[indicator_columns], columns=indicator_columns).astype(bool)
        indicators = from_indicators(indicators=upset_raw.columns, data=upset_raw)

        plot(indicators, min_degree=3, min_subset_size=10, totals_plot_elements=5, sort_by='cardinality', 
             sort_categories_by='cardinality', facecolor='deeppink', other_dots_color=0.1, shading_color=0.0,
             show_percentages=True, show_counts=False)

        plt.grid(False, which='both', axis='both', visible=False, dashes=(1, 3), zorder=4, color='gray', ls=':')

        # Title
        plt.text(-15, 100, title, fontweight='bold', fontsize=20, fontfamily='monospace')

        # Commentary
        for i, line in enumerate(commentary):
            plt.text(-14, 90 - i * 10, line, fontweight='light', fontsize=14, fontfamily='monospace')

        # Commentary circles
        bbox_props = dict(boxstyle="circle, pad=0.3", fc='#11D3CD', ec='#11D3CD')
        for pos in highlight_positions:
            plt.text(pos[0], pos[1], pos[2], ha='center', va='center', size=14, bbox=bbox_props)

        plt.show()

    def binary_balance(self, target_train, target_test, plot_dist=True):
        """
        Plot the distribution of binary target variable for both training and test datasets.
        """
        if plot_dist:
            fig, axs = plt.subplots(ncols=2, figsize=(17, 5), sharey=True)
            target_train.value_counts(normalize=True).plot(kind="bar", ax=axs[0])
            axs[0].set_title("Training Set Distribution")
            axs[0].set_xticklabels(target_train.unique(), rotation=70)
            target_test.value_counts(normalize=True).plot(kind="bar", ax=axs[1])
            axs[1].set_title("Test Set Distribution")
            axs[1].set_xticklabels(target_test.unique(), rotation=70)
            plt.show()

    def horizontal_bar(self, index, values, width=0.5, xlabel='Percentage (%)', ylabel='Label', title='Horizontal Bar Chart'):
        """
        Create a horizontal bar chart.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(y=index, width=values, height=width)

        # titles
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        plt.show()

    def multivariate_plot(self, target, feature, title):
        """
        Create a multivariate plot showing the proportion of a target variable by a feature.
        """
        fig, ax = plt.subplots()

        (self.dataframe.groupby(target)[feature]
         .value_counts(normalize=True)
         .unstack()
        ).plot(kind='bar', ax=ax)

        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        ax.set(ylabel='Proportion %', title=title)
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def kde_plot(self, x_feature, hue_feature, title):
        """
        Create a Seaborn KDE plot.
        """
        fig, g = plt.subplots(figsize=(10, 4), dpi=120)
        sns.kdeplot(data=self.dataframe, x=x_feature, hue=hue_feature,
                    common_norm=False, ec='black', shade=True,
                    alpha=0.6, lw=2, zorder=2)
        plt.title(title)
        plt.show()

    def histplot(self, x_feature, hue_feature, title):
        """
        Create a Seaborn histogram plot.
        """
        fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
        sns.histplot(data=self.dataframe, x=x_feature, hue=hue_feature, multiple="stack", ax=ax)
        ax.set(xlabel=x_feature, ylabel='Count', title=title)
        plt.show()

    def scatter_plot(self, x_feature, y_feature, hue_feature, size_feature, title):
        """
        Create a scatter plot.
        """
        avg_monthly_charge = self.dataframe[y_feature].mean()

        ax = sns.relplot(x=x_feature, y=y_feature, data=self.dataframe.sample(2100), marker="*", size=size_feature,
                         hue=hue_feature, sizes=(15, 75), kind='scatter', height=6, aspect=10/6)

        ax.set(xlabel=x_feature, ylabel=y_feature, title=title)
        plt.axhline(y=avg_monthly_charge, linewidth=2, linestyle='-.', c='crimson')
        plt.show()

    def custom_feature_plot(self, df_agg, feature, xlim=None):
        """
        Create a custom plot for a feature showing observations and default rate.
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        if xlim is not None:
            df_agg = df_agg.iloc[xlim[0]:xlim[1] + 1]

        ax1.bar(list(df_agg.index), list(df_agg["count"]))
        ax2.plot(list(df_agg["mean_target"]), color='r', linewidth=3)

        ax1.grid(False)
        ax1.set_title(feature, fontsize=20)
        ax1.set_ylabel("Observations", fontsize=14)
        ax2.set_ylabel("Target Rate (%)", fontsize=14)
        ax2.set_ylim(bottom=0)

        ax1.set_xticks(np.arange(len(df_agg.index)))
        labels = list(df_agg.index)
        ax1.axes.set_xticklabels(labels, rotation=45)
        plt.show()
