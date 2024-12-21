import os
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Функція для завантаження файлу з баром прогресу
def download_file_with_progress(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f, tqdm(
        desc=f"Завантаження {os.path.basename(save_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Файл завантажено: {save_path}")

# Завантаження та перевірка датасету
def download_and_prepare_nsl_kdd():
    dataset_url = "https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain+.txt"
    save_dir = "nsl_kdd"
    os.makedirs(save_dir, exist_ok=True)
    train_file_path = os.path.join(save_dir, "KDDTrain+.txt")

    # Перевірка, чи файл вже існує
    if not os.path.exists(train_file_path):
        print("Файл не знайдено, починаю завантаження...")
        download_file_with_progress(dataset_url, train_file_path)
    else:
        print(f"Файл вже існує: {train_file_path}")

    return train_file_path

# Завантаження, обробка та підготовка датасету
def load_and_prepare_data():
    file_path = download_and_prepare_nsl_kdd()

    # Опис колонок NSL-KDD, з урахуванням додаткових полів
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]

    # Завантаження датасету
    df = pd.read_csv(file_path, names=columns)

    # Попередня обробка: створення цільової змінної
    df['is_perl_attack'] = df['label'].apply(
        # lambda x: 1 if isinstance(x, str) and 'neptune' in x.lower() else 0
        lambda x: 1 if isinstance(x, str) and 'neptune' in x.lower() else 0
    )

    # Видалення початкових міток та "difficulty"
    df.drop(['label', 'difficulty'], axis=1, inplace=True)

    # Поділ колонок на числові та категоріальні
    categorical_columns = ['protocol_type', 'service', 'flag']
    numeric_columns = [col for col in df.columns if col not in 
                       categorical_columns + ['is_perl_attack']]

    # Поділ на вхідні дані та цільову змінну
    X = df.drop('is_perl_attack', axis=1)
    y = df['is_perl_attack']

    # Побудова пайплайну для обробки даних
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
        ]
    )

    # Застосування пайплайну до даних
    X = preprocessor.fit_transform(X)

    # Поділ на навчальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_and_prepare_data()
print(f"Розмір навчальної вибірки: {X_train.shape}")
print(f"Розмір тестової вибірки: {X_test.shape}")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

def train_and_evaluate_pnn(X_train, X_test, y_train, y_test):
    # Створення моделі PNN (на основі ймовірностей)
    pnn_model = GaussianNB()

    # Навчання моделі
    pnn_model.fit(X_train, y_train)

    # Прогнозування
    y_pred = pnn_model.predict(X_test)

    # Оцінка продуктивності моделі
    print("Звіт про класифікацію:")
    print(classification_report(y_test, y_pred))
    print(f"Точність моделі: {accuracy_score(y_test, y_pred):.2f}")

train_and_evaluate_pnn(X_train, X_test, y_train, y_test)

y_train.value_counts()

y_test.value_counts()

