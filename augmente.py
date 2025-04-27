import pandas as pd
import numpy as np

# Завантажуємо початковий датасет
df = pd.read_csv("/content/HybridModel/dataset/AllFeatureData.csv")

# Копіюємо для зміни
df_new = df.copy()

# Ідентифікуємо нормальні рядки (без атаки)
normal_rows = df_new[df_new.iloc[:, -1] == 0]

# Скільки нових атак створити
n_samples = int(0.2 * len(normal_rows))  # 20% від нормальних

# Випадково вибираємо частину нормальних даних для атаки
attack_candidates = normal_rows.sample(n=n_samples, random_state=42).copy()

# === СЦЕНАРІЇ АТАК ===

# Припустимо що координати це перші кілька ознак: X, Y, Z = ознаки 0,1,2
# Змінюємо координату X плавним дрейфом
attack_candidates.iloc[:, 0] += np.random.uniform(0.5, 2.0, size=n_samples)

# Одночасно трохи змінюємо висоту (Z, колонка 2)
attack_candidates.iloc[:, 2] += np.random.uniform(1.0, 5.0, size=n_samples)

# Невелике збільшення швидкості (припустимо швидкість на колонці 3, якщо є)
if df.shape[1] > 4:
    attack_candidates.iloc[:, 3] *= np.random.uniform(1.01, 1.05, size=n_samples)

# Мітимо їх як атаки
attack_candidates.iloc[:, -1] = 1

# Видаляємо ці рядки зі старих нормальних щоб уникнути дублів
df_new = df_new.drop(attack_candidates.index)

# Додаємо змінені рядки назад
df_augmented = pd.concat([df_new, attack_candidates], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Зберігаємо новий файл
df_augmented.to_csv("/content/HybridModel/dataset/AllFeatureData_augmented.csv", index=False)

print("Новий датасет створено: /content/HybridModel/dataset/AllFeatureData_augmented.csv")
