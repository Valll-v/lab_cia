import matplotlib.pyplot as plt  # визуализация
from sklearn.datasets import make_blobs  # генерация данных
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd


def plot_visualisation(base_range, base_list):
    plt.plot(base_range, base_list)
    plt.show()


class Clustering:
    data = []
    centers = 0

    def set_data(self, data):
        self.data = data

    def __init__(self, n_samples, centers):
        self.generate_data(n_samples, centers)

    def generate_data(self, n_samples, centers):
        self.centers = centers
        self.data = make_blobs(n_samples=n_samples, centers=centers)[0]

    def scatter_visualisation(self, c=None):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=c)
        plt.show()

    def fit(self, n):
        k_means_model = KMeans(n_clusters=n)
        k_means_model.fit(self.data)
        return k_means_model.labels_, k_means_model.inertia_

    def scan(self, eps):
        clustering = DBSCAN(eps=eps)
        return clustering.fit_predict(self.data)


def base_k_means(model):
    model.scatter_visualisation()
    list_ = []
    print('Пожалуйста подождите...')
    base_range = range(1, 10)
    for i in base_range:
        list_.append(model.fit(i)[1])
    plot_visualisation(base_range, list_)
    c = int(input('На основе графика введите оптимальное число кластеров: '))
    print('Пожалуйста подождите...')
    model.scatter_visualisation(model.fit(c)[0])


def db_scan(model: Clustering):
    print('Пожалуйста подождите...')
    model.scatter_visualisation(model.scan(float(input('Укажите максимальное удаление точки от цивилизации)'))))


def base_cycle(model):
    base_k_means(model)
    while True:
        answer = input('Готово! Желаете воспользоваться методом (Y, n)')
        if answer == 'Y':
            db_scan(model)
            break
        elif answer == 'n':
            break
        else:
            print('Неправильный формат ответа: (Y, n)')


def main():
    while True:
        try:
            model = Clustering(int(input('Введите число элементов в датасете: ')),
                               int(input('Введите число центров: ')))
            base_cycle(model)
            while True:
                filename = input('Если хотите приступить к чтению файла, введите название (иначе Enter):')
                if filename:
                    try:
                        return do_file(pd.read_csv(filename), model)
                    except FileNotFoundError:
                        print('Файл не найден')
        except ValueError as ex:
            print(f'Вы ввели неверные данные: {ex}')


def do_file(file, model: Clustering):
    data = file[['Spending Score (1-100)', 'Annual Income (k$)']].iloc[::].values
    model.set_data(data)
    return base_cycle(model)


if __name__ == '__main__':
    main()
