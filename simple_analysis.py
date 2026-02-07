import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Загрузка данных
with open('events (1).json', 'r') as f:
    data = json.load(f)

# Преобразование в DataFrame
df = pd.DataFrame(data["events"])

print("Всего событий:", len(df))
print("\nРаспределение по типам:")
print(df['signature'].value_counts())

# Визуализация
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, y="signature", order=df['signature'].value_counts().index)

plt.title("Распределение типов событий безопасности", fontsize=14, fontweight='bold')
plt.xlabel("Количество событий", fontsize=12)
plt.ylabel("Типы событий", fontsize=10)

# Добавление значений на столбцы
for p in ax.patches:
    width = p.get_width()
    plt.text(width + 0.5, p.get_y() + p.get_height()/2, 
             f'{int(width)}', ha='left', va='center')

plt.tight_layout()
plt.savefig('events_distribution_simple.png', dpi=300, bbox_inches='tight')
plt.show()