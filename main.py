import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def import_data(path: str='data.txt') -> tuple[list[float], list[float], list[float]]:
    """
    Читает файл data.txt и извлекает данные

    Parameters:
        path (str): полный или относительный путь до файла

    Returns:
        dict[str:list]: словарь со всеми данными
    """
    z, t, d = [], [], []  # Инициализация списков
    with open(path) as file:  # Открываем файл на чтение
        for i in [x.strip() for x in file]:  # Идём по строкам
            x = i.split()  # Разделяем по пробелам строку
            z.append(float(x[0]))  # Добавляем глубину, преобразуя в float
            t.append(float(x[1]))  # Добавляем температуру, преобразуя в float
            d.append(float(x[2]))  # Добавляем производную, преобразуя в float
    return z, t, d


def get_params() -> tuple[float, int]:
    """
    Ввод необходимых для алгоритма данных

    Returns:
        dict[str:float | int]: словарь с альфа и окном
    """
    alpha = float(input('Alpha экспоненциального сглаживания: '))
    window = int(input('Окно сглаживания скользящего среднего: '))
    if alpha < 0 or alpha > 1:
        raise ValueError('Alpha должна быть в пределах от 0 до 1')
    if window < 2 or window > 13125:
        raise ValueError('Окно не должно быть больше количества элементов или меньше 2')
    return alpha, window


def draw_plots(
        z: list[float],
        t: list[float],
        alpha: float,
        window: int
) -> None:
    """
    Обработка данных и отрисовка графиков

    Parameters:
        z (list[float]): массив данных глубины
        t (list[float]): массив данных температуры
        alpha (float): коэффициент альфа экспоненциального сглаживания
        window (int): окно для сглаживания скользящим средним
    """
    df = pd.DataFrame(dict(z=z, T=t))
    plots = {
        'Изменение температуры от глубины': df,
        f'Экспоненциальное сглаживание, alpha = {alpha}': df.ewm(alpha=alpha).mean(),
        f'Сглаживание скользящим средним, окно = {window}': df.rolling(window=window).mean()
    }
    for title, data in plots.items():
        data.plot(x='z', y='T', kind='line')
        plt.title(title)
        plt.xlabel('Глубина')
        plt.ylabel('Температура')
        plt.grid()
        plt.show()


def draw_derivative(z: list[float], d: list[float]) -> None:
    """
    Отрисовка графика производной

    Parameters:
        z (list[float]): массив данных глубины
        d (list[float]): массив данных производной
    """
    plt.plot(z, d)
    plt.title('Производная, dT/dz')
    plt.xlabel('Глубина')
    plt.ylabel('Производная')
    plt.grid()
    plt.show()


def get_derivative(
        x1: float,
        x2: float,
        z: list[float],
        derivative: list[float]
) -> None:
    """
    Расчёт среднего арифметического производной в промежутке [x1;x2]

    Parameters:
        x1 (float): начальная точка абсциссы
        x2 (float): конечная точка абсциссы
        z (list[float]): массив с данными глубины
        derivative (list[float]): массив с данными производной
    """
    statements = [
        x1 >= x2,
        x1 < min(z),
        x1 > max(z),
        x2 < min(z),
        x2 > max(z),
    ]
    if statements.count(True) > 0:
        raise ValueError('Неверно заданные x1 или x2')
    if x1 in z:
        index_x1 = z.index(x1)
    else:
        temp = [abs(x-x1) for x in z]
        index_x1 = temp.index(min(temp))
    if x2 in z:
        index_x2 = z.index(x2)
    else:
        temp = [abs(x-x2) for x in z]
        index_x2 = temp.index(min(temp))
    drv = sum(derivative[index_x1:index_x2]) / len(derivative[index_x1:index_x2])
    print(f'Производная на промежутке [{x1};{x2}]: {drv}')



def predict_trand(z: list[float], t: list[float]) -> list[float]:
    """
    Расчёт тренда, отклонения от него и отрисовка графиков

    Parameters:
        z (list[float]): массив данных глубины
        t (list[float]): массив данных температуры

    Returns:
        list[float]: массив с данными отклонения от тренда
    """
    trand_x, trand_y, deflection = [], [], t[:9]
    for i in range(9, len(z)-1):
        x = np.array(z[:i]).reshape(-1, 1)
        y = np.array(t[:i])
        model = LinearRegression()
        model.fit(x, y)
        a = model.intercept_
        b = model.coef_[0]
        future_x = np.array([z[i+1]]).reshape(-1, 1)
        future_y = model.predict(future_x)
        trand_x.append(future_x[0])
        trand_y.append(future_y[0])
        deflection.append(float(t[i+1]-future_y[0]))
    plt.plot(z, t, color='blue', label='Исходные данные')
    plt.plot(trand_x, trand_y, color='red', label='Прогнозируемый тренд')
    plt.plot(z[:-1], deflection, color='green', label='Отклонение от тренда')
    plt.xlabel('Глубина')
    plt.ylabel('Температура')
    plt.legend()
    plt.show()
    return deflection


def get_anomalies(z: list[float], deflection: list[float]) -> None:
    """
    Поиск промежутков из данных по отклонения

    Parameters:
        z (list[float]): массив данных глубины
        deflection (list[float]): массив с отклонениями от тренда
    """
    trand = [1 if x >= 5 else 0 for x in deflection]
    anomalies = []
    start, stop = 0, 0
    for p in range(len(trand)-1):
        if trand[p] == 1 and trand[p-1] == 0:
            start = p
        elif trand[p] == 1 and (trand[p+1] == 0 or p == len(trand)-1):
            stop = p
            anomalies.append((start, stop))
    filtered = []
    for a in range(len(anomalies)-1):
            start1, stop1 = anomalies[a]
            start2, stop2 = anomalies[a+1]
            if abs(stop1 - start2) < 11:
                filtered.append((start1, stop2))
            else:
                filtered.append((start1, stop1))
    anomalies, ban = set(), set()
    for f in range(len(filtered)-1):
        start1, stop1 = filtered[f]
        start2, stop2 = filtered[f+1]
        if stop1 in range(start2, stop2+1):
            anomalies.add((start1, stop2))
            ban.add((start2, stop2))
        else:
            anomalies.add((start1, stop1))
    result = 'Промежутки отклонения данных: '
    for x1, x2 in anomalies.difference(ban):
        result += f'[{z[x1]};{z[x2]}], '
    print(result[:-2])


if __name__ == '__main__':
    z, t, d = import_data()
    alpha, window = get_params()
    draw_plots(z, t, alpha, window)
    draw_derivative(z, d)
    get_derivative(
        float(input('Начальная точка (x1): ')),
        float(input('Конечная точка (x2): ')),
        z,
        d
    )
    print('Процесс расчёта тренда, пожалуйста, ожидайте.')
    deflection = predict_trand(z, t)
    get_anomalies(z, deflection)
