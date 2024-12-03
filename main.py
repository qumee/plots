import plotly.express as px
import pandas as pd


def import_data() -> tuple[list[float], list[float]]:
    """
    Читает файл data.txt и извлекает z и T
    :return:
        z (list[float]): z - отсортированная глубина
        t (list[float]): T - отсортированная температура
    """
    z, t = [], [] # Инициализация списков
    with open('data.txt') as file:  # Открываем файл на чтение
        for i in [x.strip() for x in file]:  # Идём по строкам
            x = i.split()  # Разделяем по пробелам строку
            z.append(float(x[0]))  # Добавляем в список z-ов, преобразуя в float
            t.append(float(x[1]))  # Добавляем в список t-ов, преобразуя в float
    return z, t


def draw(
        z: list,
        t: list,
        alpha: float,
        window: int
) -> None:
    """
    Отрисовывает по полученным данным графики
    :param z: список z (глубина)
    :param t: список t (температура)
    :param alpha: коэффициент alpha экспоненциального сглаживания
    :param window: окно сглаживания скользящего среднего
    """
    df = pd.DataFrame(  # Датафрейм pandas - работает с данными
        dict(
            z=z,
            T=t
        )
    )
    titles = [  # Надписи для графиков
        'Исходные данные',
        'Экспоненциальное сглаживание',
        'Сглаживание скользящим средним'
    ]
    j = 0  # Переменная-итератор для надписей
    for plot in [
        df,  # Исходные данные
        df.ewm(alpha=alpha).mean(),  # Экспоненциальное сглаживание
        df.rolling(window=window).mean()  # Сглаживание скользящим средним
    ]:
        plot.sort_values(by='z')  # Сортирует по глубине
        fig = px.line(  # Инициализация класса линии для графика
            data_frame=plot,
            x="z",
            y="T",
            title=titles[j]
        )
        fig.show()  # Отрисовка графика
        j += 1


def enter() -> tuple[float, int]:
    """
    Функция ввода данных, проверяет на соответствие
    :return:
        alpha (float): Коэффициент alpha экспоненциального сглаживания
        window (int): Окно сглаживания скользящего среднего
    """
    alpha = float(input('Alpha экспоненциального сглаживания: '))
    window = int(input('Окно сглаживания скользящего среднего: '))
    if alpha < 0 or alpha > 1:
        raise ValueError('Alpha должна быть в пределах от 0 до 1')
    if window < 2 or window > 13125:
        raise ValueError('Окно не должно быть больше количества элементов или меньше 2')
    return alpha, window


if __name__ == '__main__':
    draw(
        *import_data(),
        *enter()
    )