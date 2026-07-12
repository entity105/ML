from matplotlib import pyplot as plt
import numpy as np

def create_window(*args, figsize=(10, 6), title=None, ox_name='x', oy_name='y') -> tuple:
    """Создаёт окно с 2D осями (n штук)"""
    # Написать инструкцию
    fig, axs = plt.subplots(*args, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    if len(args) == 0:
        axs = [axs, ]
    for ax in axs:
        ax.set_xlabel(ox_name)
        ax.set_ylabel(oy_name)
        ax.grid(True, alpha=0.3)
    return fig, *axs



def drow2Dgraph(axis, x, y,
                line='b-', linewidth=2, name_func=None,
                ox_name='x', oy_name='y',
                is_show=False) -> None:
    """Рисует один график в окне"""

    ax = axis
    ax.plot(x, y, line, linewidth=linewidth, label=name_func)
    ax.set_xlabel(ox_name)
    ax.set_ylabel(oy_name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if is_show:
        plt.show()

def generate_formula(features: list, w: list, left_part='a(x)') -> str:
    # Собираем слагаемые
    formula_parts = [f'{coeff:.2f}' if term == '1' else f'{coeff:.2f}{term}' for coeff, term in zip(w, features)]
    formula = ' + '.join(formula_parts)
    return f'${left_part} = {formula}$'

def approximation(x, y, func_str: str,
                  features: list, w, Q: float):
    formula = generate_formula(features, w)
    fig, ax = create_window(title='Аппроксимация функции')
    drow2Dgraph(ax, x, y, name_func=func_str)
    # plt.ion()
    model_line, = ax.plot([], [], 'r-', linewidth=2, label=formula)
    text = ax.text(
        0.02, 0.95,  # 1. Координаты (x, y)
        'Итерация: 0\n'
        f'Q = {Q}',  # 2. Текст
        transform=ax.transAxes,  # 3. Система координат
        fontsize=12,  # 4. Размер шрифта
        verticalalignment='top',  # 5. Вертикальное выравнивание
        bbox=dict(  # 6. Подложка (рамка)
            boxstyle='round',  # — закруглённые углы
            facecolor='white',  # — цвет фона
            alpha=0.8  # — прозрачность (0.8 = почти непрозрачный)
        )
    )
    return ax, model_line, text

def approximation_update(ax, x, w, model, model_line,
                         features: list, text, iteration, Q, pause=0.02):
    y_model = model(w, x)
    model_line.set_data(x, y_model)
    model_line.set_label(generate_formula(features, w))
    ax.legend()
    text.set_text(f'Итерация: {iteration + 1}\nQ = {Q:.4f}')
    plt.pause(pause)

def drow_cls_points(ax, x_train, y_train):
    # Добавить проверки на массивы
    # Сделать многоклассовую клас.
    x_0 = x_train[y_train == -1]
    x_1 = x_train[y_train == 1]
    ax.scatter(x_0[:, 1], x_0[:, 2], color='red')
    ax.scatter(x_1[:, 1], x_1[:, 2], color='blue')

def create_line_2D(ax, features, left_part):
    w_norm = [0, 0]
    formula = generate_formula(features, w_norm, left_part=left_part)
    line, = ax.plot([], [], label=formula)
    ax.legend()
    return line

def add_text(ax, text:str, x=0.02, y=0.95):
    text = ax.text(
        x, y,  # 1. Координаты (x, y)
        text,  # 2. Текст
        transform=ax.transAxes,  # 3. Система координат
        fontsize=12,  # 4. Размер шрифта
        verticalalignment='top',  # 5. Вертикальное выравнивание
        bbox=dict(  # 6. Подложка (рамка)
            boxstyle='round',  # — закруглённые углы
            facecolor='white',  # — цвет фона
            alpha=0.8  # — прозрачность (0.8 = почти непрозрачный)
        )
    )
    return text

def update_line_2D(line, edges:tuple, w, features:list, left_part:str):
    w_norm = [w[1] / w[2], w[0] / w[2]]
    updated_y = [-x * w_norm[0] - w_norm[1] for x in edges]
    line.set_data(edges, updated_y)
    line.set_label(
         generate_formula(features, w_norm, left_part=left_part)
    )


class MovePoint:
    def __init__(self, title:str, axis_place:tuple[int, int]=(), *args, **kwargs):
        """Создаёт окно с заголовком и оси. *args, **kwargs - остальные аргументы plt.subplots()"""
        fig, axes = plt.subplots(*axis_place, *args, **kwargs)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        self.fig = fig
        self.axes = [axes, ] if axis_place and axis_place[0] == 1 else axes     # Либо матрица, либо скаляр

        self.texts_fig_to_update = []
        self.texts_ax_to_update = []
        self.points = []

    def set_default_text_fig(self, n:int, formula:str):
        """Добавляет стандартный текст в окно"""
        y_place = 0.9
        itr_str = self.fig.text(0.2, y_place, f'Итерация: 0 / {n}', ha='center', fontsize=14)
        self.fig.text(0.4, y_place, formula, fontsize=14, fontweight='bold')
        self.fig.subplots_adjust(top=y_place - 0.02, bottom=0.1)
        self.texts_fig_to_update.append(itr_str)

    def set_text_fig(self, text:str, coord_text:tuple[float, float]=(0.4, 0.9), to_updata=False, *args, **kwargs):
        """Пользовательский текст для окна, to_updata=True позволяет далее обновлять его"""
        text_obj = self.fig.text(*coord_text, text, *args, **kwargs)
        if to_updata:
            self.texts_fig_to_update.append(text_obj)
        self.fig.subplots_adjust(top=coord_text[1] - 0.02, bottom=0.1)

    def update_fig_text(self, idx:int, new_text:str):
        """Обновить оконный текст по индексу в списке"""
        self.texts_fig_to_update[idx].set_text(new_text)

    def set_text_axis(self, text:str, ax_idx:tuple[int, int]=None, coord_text:tuple[float, float]=(0.1, 0.95), to_updata=False, base_setting=True, **kwargs):
        """Пользовательский текст для оси по индексу ax_idx (если осей > 1), to_updata=True позволяет далее обновлять его"""
        selected_axis = self.select_axis(ax_idx)

        if base_setting:
            kwargs = {'transform': selected_axis.transAxes,
                      'fontsize' : 12,
                      'verticalalignment' : 'top',
                      'bbox' : dict(  # 6. Подложка (рамка)
                            boxstyle='round',  # — закруглённые углы
                            facecolor='white',  # — цвет фона
                            alpha=0.8  # — прозрачность (0.8 = почти непрозрачный)
                      )
            }

        text_obj = selected_axis.text(*coord_text, text, **kwargs)
        if to_updata:
            self.texts_ax_to_update.append(text_obj)

    def updata_ax_text(self, idx, new_text:str):
        """Обновить текст на осях по индексу в списке"""
        self.texts_ax_to_update[idx].set_text(new_text)

    def drow_graph(self, x_data, y_data, ax_idx:tuple[int, int]=None, base_setting=True, axis_name=('x', 'y'), **setting):
        """Делает график на оси по индексу ax_idx (если осей > 1)"""
        selected_axis = self.select_axis(ax_idx)

        if base_setting:
            setting = dict(color='blue', linewidth=3, label=None)

        selected_axis.plot(x_data, y_data, **setting)
        selected_axis.set_xlabel(axis_name[0])
        selected_axis.set_ylabel(axis_name[1])
        selected_axis.grid(True, alpha=0.3)

    def make_point(self, x_data, y_data, ax_idx:tuple[int, int]=None, base_setting=True, to_update=False, *settings_pos, **settings_names):
        """Делает точку на оси по индексу ax_idx (если осей > 1)"""
        selected_axis = self.select_axis(ax_idx)

        if base_setting:
            settings_pos = ('ro', )
            settings_names = dict(markersize=12)

        point = selected_axis.plot([x_data], [y_data], *settings_pos, **settings_names)
        if to_update:
            self.points.append(*point)

    def updata_point(self, new_x, new_y, idx:int):
        """Обновляет точку по индексу idx в списке"""
        self.points[idx].set_data([new_x], [new_y])

    def select_axis(self, ax_idx:tuple[int, int]=None):
        """Принимает двумерный индекс, возвращает ось"""
        i, j = ax_idx
        try:
            selected_axis = self.axes[i][j] if ax_idx else self.axes
        except IndexError:
            raise IndexError(f"Оси с координатами {ax_idx} не существует")
        return selected_axis


class DynamicGraphs:
    def __init__(self, title:str, axis_place:tuple[int, int]=(), *args, **kwargs):
        """Создаёт окно с заголовком и оси. *args, **kwargs - остальные аргументы plt.subplots()"""
        fig, axes = plt.subplots(*axis_place, *args, **kwargs)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        self.fig = fig
        self.axes = [axes, ] if axis_place and axis_place[0] == 1 else axes  # Либо матрица, либо скаляр

        self.texts = []
        self.graphs = []

    def set_default_text_fig(self, n:int, formula:str):
        """Добавляет стандартный текст в окно"""
        y_place = 0.9
        itr_str = self.fig.text(0.2, y_place, f'Итерация: 0 / {n}', ha='center', fontsize=14)
        self.fig.text(0.4, y_place, formula, fontsize=14, fontweight='bold')
        self.fig.subplots_adjust(top=y_place - 0.02, bottom=0.1)
        self.texts.append(itr_str)

    def set_text_fig(self, text:str, coord_text:tuple[float, float]=(0.4, 0.9), to_updata=False, *args, **kwargs):
        """Пользовательский текст для окна, to_updata=True позволяет далее обновлять его"""
        text_obj = self.fig.text(*coord_text, text, *args, **kwargs)
        if to_updata:
            self.texts.append(text_obj)
        self.fig.subplots_adjust(top=coord_text[1] - 0.02, bottom=0.1)

    def update_text(self, idx:int, new_text:str):
        """Обновить текст по индексу в списке"""
        self.texts[idx].set_text(new_text)

    def set_text_axis(self, text:str, ax_idx:tuple[int, int]=None, coord_text:tuple[float, float]=(0.1, 0.95), to_updata=False, base_setting=True, **kwargs):
        """Пользовательский текст для оси по индексу ax_idx (если осей > 1), to_updata=True позволяет далее обновлять его"""
        selected_axis = self.select_axis(ax_idx)

        if base_setting:
            kwargs = {'transform': selected_axis.transAxes,
                      'fontsize' : 12,
                      'verticalalignment' : 'top',
                      'bbox' : dict(  # 6. Подложка (рамка)
                            boxstyle='round',  # — закруглённые углы
                            facecolor='white',  # — цвет фона
                            alpha=0.8  # — прозрачность (0.8 = почти непрозрачный)
                      )
            }

        text_obj = selected_axis.text(*coord_text, text, **kwargs)
        if to_updata:
            self.texts.append(text_obj)

    def drow_graph(self, x_data, y_data, ax_idx:tuple[int, int]=None, base_setting=True, to_updata=False, axis_name=('x', 'y'), **setting):
        """Делает график на оси по индексу ax_idx (если осей > 1)"""
        selected_axis = self.select_axis(ax_idx)

        if base_setting:
            setting = dict(color='blue', linewidth=3, label=None)

        graph = selected_axis.plot(x_data, y_data, **setting)
        selected_axis.set_xlabel(axis_name[0])
        selected_axis.set_ylabel(axis_name[1])
        selected_axis.grid(True, alpha=0.3)
        if to_updata:
            self.graphs.append(*graph)
        selected_axis.legend()

    def updata_graphs(self, x_data, y_data, idx:int=None, new_label=None):
        graph_obj = self.graphs[idx] if idx else self.graphs[0]
        graph_obj.set_data(x_data, y_data)
        if new_label:
            graph_obj.set_label(new_label)
            self.axes.legend()


    def select_axis(self, ax_idx:tuple[int, int]=None):
        """Принимает двумерный индекс, возвращает ось"""
        try:
            selected_axis = self.axes[ax_idx[0]][ax_idx[1]] if ax_idx else self.axes
        except IndexError:
            raise IndexError(f"Оси с координатами {ax_idx} не существует")
        return selected_axis