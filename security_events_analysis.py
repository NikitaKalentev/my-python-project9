import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter

# ================== Этап 1: Подготовка данных ==================

def load_json_data(file_path):
    """Загружает данные из JSON файла в DataFrame"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Преобразуем в DataFrame
        df = pd.DataFrame(data['events'])
        
        print("✅ Данные успешно загружены!")
        print(f"📊 Всего событий: {len(df)}")
        print(f"📅 Период данных: от {df['timestamp'].min()} до {df['timestamp'].max()}")
        print("\nПервые 5 строк данных:")
        print(df.head())
        
        # Преобразуем timestamp в datetime для удобства
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Файл {file_path} не найден!")
        return None
    except json.JSONDecodeError:
        print("❌ Ошибка чтения JSON файла!")
        return None
    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        return None

# ================== Этап 2: Анализ данных ==================

def analyze_signatures(df):
    """Анализирует распределение событий по типам (signature)"""
    
    if df is None or df.empty:
        print("❌ Нет данных для анализа!")
        return None
    
    print("\n" + "="*60)
    print("📈 АНАЛИЗ РАСПРЕДЕЛЕНИЯ СОБЫТИЙ ПО ТИПАМ")
    print("="*60)
    
    # Подсчет частоты каждого типа события
    signature_counts = df['signature'].value_counts()
    
    print(f"\n🔢 Всего уникальных типов событий: {len(signature_counts)}")
    print("\n📋 Распределение событий:")
    for i, (signature, count) in enumerate(signature_counts.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"  {i:2d}. {signature[:60]:60} : {count:3d} событий ({percentage:.1f}%)")
    
    # Статистика
    print("\n📊 СТАТИСТИКА:")
    print(f"  • Самый частый тип: '{signature_counts.index[0]}' ({signature_counts.iloc[0]} событий)")
    print(f"  • Среднее количество событий на тип: {signature_counts.mean():.1f}")
    print(f"  • Медиана: {signature_counts.median()}")
    print(f"  • Стандартное отклонение: {signature_counts.std():.1f}")
    
    return signature_counts

# ================== Этап 3: Визуализация данных ==================

def create_visualizations(df, signature_counts):
    """Создает визуализации распределения событий"""
    
    if df is None or df.empty:
        print("❌ Нет данных для визуализации!")
        return
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # ===== График 1: Bar plot (основной) =====
    plt.figure(figsize=(14, 8))
    
    # Укорачиваем длинные названия для лучшей читаемости
    short_labels = []
    for sig in signature_counts.index:
        # Берем только первую часть до первого пробела или первые 30 символов
        parts = sig.split()
        if len(parts) > 1:
            short_labels.append(parts[0] + " " + parts[1][:20])
        else:
            short_labels.append(sig[:30] + "...")
    
    bars = plt.bar(range(len(signature_counts)), signature_counts.values, 
                   color=plt.cm.Set3(range(len(signature_counts))))
    
    # Настройки графика
    plt.title('Распределение событий информационной безопасности по типам', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Типы событий', fontsize=12)
    plt.ylabel('Количество событий', fontsize=12)
    plt.xticks(range(len(signature_counts)), short_labels, rotation=45, ha='right')
    
    # Добавление значений на столбцы
    for i, (bar, count) in enumerate(zip(bars, signature_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('security_events_distribution.png', dpi=300, bbox_inches='tight')
    print("\n📊 График 1 сохранен как 'security_events_distribution.png'")
    
    # ===== График 2: Pie chart (процентное соотношение) =====
    plt.figure(figsize=(10, 10))
    
    # Берем топ-8 типов, остальные объединяем в "Другие"
    if len(signature_counts) > 8:
        top_8 = signature_counts.head(8)
        others = pd.Series([signature_counts[8:].sum()], index=['Другие'])
        pie_data = pd.concat([top_8, others])
    else:
        pie_data = signature_counts
    
    # Автоматические цвета
    colors = plt.cm.Set3(range(len(pie_data)))
    
    plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 10})
    plt.title('Процентное распределение событий по типам', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('security_events_pie_chart.png', dpi=300, bbox_inches='tight')
    print("📈 График 2 сохранен как 'security_events_pie_chart.png'")
    
    # ===== График 3: Timeline событий =====
    plt.figure(figsize=(15, 6))
    
    # Группируем по часам
    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df.groupby('hour').size()
    
    plt.plot(hourly_counts.index, hourly_counts.values, marker='o', 
             linewidth=2, markersize=8)
    plt.fill_between(hourly_counts.index, hourly_counts.values, alpha=0.3)
    
    plt.title('Распределение событий по времени суток', fontsize=14, fontweight='bold')
    plt.xlabel('Час дня', fontsize=12)
    plt.ylabel('Количество событий', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('security_events_timeline.png', dpi=300, bbox_inches='tight')
    print("⏰ График 3 сохранен как 'security_events_timeline.png'")
    
    # ===== График 4: Countplot (используя Seaborn) =====
    plt.figure(figsize=(12, 6))
    
    # Создаем сокращенные подписи для оси X
    df['signature_short'] = df['signature'].apply(
        lambda x: ' '.join(x.split()[:2])[:25] + '...' if len(x) > 25 else x
    )
    
    ax = sns.countplot(data=df, y='signature_short', order=df['signature_short'].value_counts().index,
                      palette='viridis')
    
    plt.title('Распределение типов событий безопасности (Seaborn)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Количество событий', fontsize=12)
    plt.ylabel('Типы событий', fontsize=12)
    
    # Добавление значений на столбцы
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.5, p.get_y() + p.get_height()/2,
                f'{int(width)}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('security_events_seaborn.png', dpi=300, bbox_inches='tight')
    print("🎨 График 4 сохранен как 'security_events_seaborn.png'")
    
    print("\n✅ Все графики успешно созданы и сохранены!")
    
    # Показываем все графики
    plt.show()

# ================== Основная программа ==================

def main():
    print("="*60)
    print("📊 АНАЛИЗ СОБЫТИЙ ИНФОРМАЦИОННОЙ БЕЗОПАСНОСТИ")
    print("="*60)
    
    # Путь к файлу с данными
    json_file = 'events (1).json'
    
    # Этап 1: Загрузка данных
    df = load_json_data(json_file)
    
    if df is None:
        return
    
    # Этап 2: Анализ данных
    signature_counts = analyze_signatures(df)
    
    # Этап 3: Визуализация
    if signature_counts is not None:
        create_visualizations(df, signature_counts)
        
        # Дополнительная информация
        print("\n" + "="*60)
        print("📋 ИНФОРМАЦИЯ ДЛЯ ОТЧЕТА")
        print("="*60)
        print(f"• Проанализировано событий: {len(df)}")
        print(f"• Уникальных типов событий: {len(signature_counts)}")
        print(f"• Период анализа: {df['timestamp'].min().date()} - {df['timestamp'].max().date()}")
        print(f"• Самый частый тип события: '{signature_counts.index[0]}'")
        print(f"• Создано графиков: 4")
        print("• Файлы сохранены в текущей директории")

# Запуск программы
if __name__ == "__main__":
    main()
