import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 전처리
# =====================================
# Load CSV file (Adjust file_path as necessary)
file_path = r"C:\Users\younj\Desktop\4-2\dataVisualization\WHR2024.csv"
df = pd.read_csv(file_path)

# 컬럼명 변환
df.columns = [
    'Country', 'Happiness_Score', 'upper_whisker', 'lower_whisker',
    'GDP', 'Social_Support', 'Health', 'Freedom', 'Generosity', 'Corruption',
    'Residual'
]

# 한국 요소 탐지
target_country = 'South Korea'
korea_data = df[df['Country'] == target_country].iloc[0]
factors = ['GDP', 'Social_Support', 'Health', 'Freedom', 'Generosity', 'Corruption']
global_avg = df[factors].mean()



# 그래프 생성
# =====================================

# 1. 행복 점수 상위 15개국
top_15 = df.sort_values(by='Happiness_Score', ascending=False).head(15)
plt.figure(figsize=(10, 6))
# Using hue and setting legend=False to prevent seaborn warnings
sns.barplot(x='Country', y='Happiness_Score', data=top_15, palette='viridis', hue='Country', legend=False)
plt.title('Top 15 Happiest Countries (2024)')
plt.xlabel('Country Name')
plt.ylabel('Happiness Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./img/1_top15.png')
plt.close()

# 2. 글로벌 행복 점수 분포표(히스토그램)
plt.figure(figsize=(8, 5))
sns.histplot(df['Happiness_Score'], kde=True, bins=15, color='skyblue')
plt.title('Distribution of Global Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Number of Countries')
plt.tight_layout()
plt.savefig('./img/2_global_distribution.png')
plt.close()

# 3. 요소에 따른 히트맵
cols_to_correlate = ['Happiness_Score'] + factors
corr_matrix = df[cols_to_correlate].corr()
plt.figure(figsize=(8, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('3.Correlation Heatmap of Happiness Factors')
plt.tight_layout()
plt.savefig('./img/3_correlation_heatmap.png')
plt.close()

# 4. gdp에 따른 행복점수
plt.figure(figsize=(9, 6))
sns.regplot(x='GDP', y='Happiness_Score', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Relationship between Log GDP per Capita and Happiness Score')
plt.xlabel('Log GDP per Capita (Contribution)')
plt.ylabel('Happiness Score')
plt.tight_layout()
plt.savefig('./img/4_gdp_vs_happiness.png')
plt.close()

# 5. 건강에 따른 행복점수
plt.figure(figsize=(9, 6))
sns.regplot(x='Health', y='Happiness_Score', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'green'})
plt.title('Relationship between Healthy Life Expectancy and Happiness Score')
plt.xlabel('Healthy Life Expectancy (Contribution)')
plt.ylabel('Happiness Score')
plt.tight_layout()
plt.savefig('./img/5_health_vs_happiness.png')
plt.close()

# 6. Top 10 vs. Bottom 10 Average Factor Contribution (Grouped Bar Chart)
top_10 = df.sort_values(by='Happiness_Score', ascending=False).head(10)[factors].mean()
bottom_10 = df.sort_values(by='Happiness_Score', ascending=True).head(10)[factors].mean()
comparison_df_avg = pd.DataFrame({'Top 10 Avg': top_10, 'Bottom 10 Avg': bottom_10})

comparison_df_avg.plot(kind='bar', figsize=(12, 6), color=['darkgreen', 'darkred'])
plt.title('Average Factor Contribution: Top 10 vs. Bottom 10')
plt.ylabel('Average Contribution')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Group')
plt.tight_layout()
plt.savefig('./img/6_top_bottom_comparison.png')
plt.close()

# 7. Freedom vs. Perceptions of Corruption (Scatter Plot)
plt.figure(figsize=(9, 6))
sns.scatterplot(x='Freedom', y='Corruption', data=df, hue='Happiness_Score', size='Happiness_Score', palette='coolwarm', sizes=(20, 200))
plt.title('Freedom vs. Perceptions of Corruption')
plt.xlabel('Freedom (Contribution)')
plt.ylabel('Perceptions of Corruption (Contribution)')
plt.legend(title='Happiness Score')
plt.tight_layout()
plt.savefig('./img/7_freedom_corruption.png')
plt.close()

# 8. Generosity Distribution (Box Plot)
plt.figure(figsize=(8, 6))
# Using x with a constant value to prevent seaborn warnings
sns.boxplot(y=df['Generosity'], x=pd.Series(['Generosity'] * len(df)), color='lightcoral')
plt.title('Distribution of Generosity Contributions')
plt.xlabel('')
plt.ylabel('Generosity Contribution')
plt.tight_layout()
plt.savefig('./img/8_generosity_boxplot.png')
plt.close()


# ==============================================================================
# 3. South Korea-Focused Analysis and Visualization (4 charts)
# ==============================================================================

continent_map = {
    # Europe
    'Finland':'Europe','Denmark':'Europe','Iceland':'Europe','Sweden':'Europe','Netherlands':'Europe',
    'Norway':'Europe','Luxembourg':'Europe','Switzerland':'Europe','Austria':'Europe','Belgium':'Europe',
    'Ireland':'Europe','Czechia':'Europe','Lithuania':'Europe','United Kingdom':'Europe','Slovenia':'Europe',
    'Romania':'Europe','Estonia':'Europe','Poland':'Europe','Spain':'Europe','Serbia':'Europe',
    'Malta':'Europe','Italy':'Europe','Slovakia':'Europe','Latvia':'Europe','Cyprus':'Europe',
    'France':'Europe','Germany':'Europe','Greece':'Europe','Bosnia and Herzegovina':'Europe',
    'Kosovo':'Europe','Hungary':'Europe','Croatia':'Europe','Moldova':'Europe','Russia':'Europe',
    'Montenegro':'Europe','Bulgaria':'Europe','Armenia':'Europe','North Macedonia':'Europe',
    'Albania':'Europe','Georgia':'Europe','Ukraine':'Europe','Turkey':'Europe',

    # Oceania
    'Australia':'Oceania','New Zealand':'Oceania',

    # North America
    'Canada':'North America','United States':'North America','Mexico':'North America',
    'Costa Rica':'North America','Panama':'North America','Guatemala':'North America',
    'Nicaragua':'North America','Jamaica':'North America','Dominican Republic':'North America',
    'Honduras':'North America','El Salvador':'North America',

    # South America
    'Uruguay':'South America','Chile':'South America','Brazil':'South America','Argentina':'South America',
    'Paraguay':'South America','Peru':'South America','Bolivia':'South America','Ecuador':'South America',
    'Colombia':'South America','Venezuela':'South America',

    # Middle East
    'Israel':'Middle East','Kuwait':'Middle East','United Arab Emirates':'Middle East','Saudi Arabia':'Middle East',
    'Bahrain':'Middle East','Iran':'Middle East','Iraq':'Middle East','Azerbaijan':'Middle East',
    'Jordan':'Middle East','Lebanon':'Middle East','State of Palestine':'Middle East','Turkey':'Middle East',

    # Asia
    'Singapore':'Asia','Taiwan Province of China':'Asia','Japan':'Asia','South Korea':'Asia','Philippines':'Asia',
    'Vietnam':'Asia','Thailand':'Asia','Malaysia':'Asia','China':'Asia','Kazakhstan':'Asia',
    'Uzbekistan':'Asia','Kyrgyzstan':'Asia','Mongolia':'Asia','Indonesia':'Asia','Hong Kong S.A.R. of China':'Asia',
    'Armenia':'Asia','Nepal':'Asia','Laos':'Asia','Cambodia':'Asia','Myanmar':'Asia','Sri Lanka':'Asia',
    'Bangladesh':'Asia','India':'Asia','Pakistan':'Asia','Georgia':'Asia','Afghanistan':'Asia','Yemen':'Asia',

    # Africa
    'South Africa':'Africa','Algeria':'Africa','Libya':'Africa','Mauritius':'Africa','Gabon':'Africa',
    'Ivory Coast':'Africa','Guinea':'Africa','Senegal':'Africa','Nigeria':'Africa','Cameroon':'Africa',
    'Namibia':'Africa','Morocco':'Africa','Niger':'Africa','Burkina Faso':'Africa','Mauritania':'Africa',
    'Gambia':'Africa','Chad':'Africa','Kenya':'Africa','Tunisia':'Africa','Benin':'Africa','Uganda':'Africa',
    'Ghana':'Africa','Liberia':'Africa','Mali':'Africa','Madagascar':'Africa','Togo':'Africa','Ethiopia':'Africa',
    'Tanzania':'Africa','Comoros':'Africa','Zambia':'Africa','Eswatini':'Africa','Malawi':'Africa',
    'Botswana':'Africa','Zimbabwe':'Africa','Congo (Brazzaville)':'Africa','Congo (Kinshasa)':'Africa',
    'Sierra Leone':'Africa','Lesotho':'Africa'
}

df['Continent'] = df['Country'].map(continent_map).fillna('Other')

# 9. Continent-wise Average Happiness Score
continent_mean = df.groupby('Continent')['Happiness_Score'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=continent_mean.index, y=continent_mean.values, palette='viridis')
plt.title('Average Happiness Score by Continent')
plt.xlabel('Continent')
plt.ylabel('Average Happiness Score')
plt.tight_layout()
plt.savefig('./img/9_continent_mean.png')
plt.close()


# 10. Stacked Bar Chart for Top 10 Countries' Factor Contributions
top10_factors = df.sort_values(by='Happiness_Score', ascending=False).head(10)
top10_factors_plot = top10_factors.set_index('Country')[factors]

top10_factors_plot.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='viridis')
plt.title('Factor Contributions of Top 10 Happiest Countries')
plt.ylabel('Contribution Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./img/10_top10_stacked.png')
plt.close()

