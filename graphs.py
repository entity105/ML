from matplotlib import pyplot as plt
import numpy as np

class Init:
    """Базовый класс"""

    def __init__(self, title:str, axis_place:tuple[int, int]=(), *args, **kwargs):
        """Создаёт окно с заголовком и оси. *args, **kwargs - остальные аргументы plt.subplots()"""
        fig, axes = plt.subplots(*axis_place, *args, **kwargs)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        self.fig = fig
        self.axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]     # Всегда список

        self.texts = []

    def set_default_text_fig(self, n:int, formula:str):
        """Добавляет стандартный текст в окно"""
        y_place = 0.9
        itr_str = self.fig.text(0.2, y_place, f'Итерация: 0 / {n}', ha='center', fontsize=14)
        self.fig.text(0.4, y_place, formula, fontsize=14, fontweight='bold')
        self.fig.subplots_adjust(top=y_place - 0.02, bottom=0.1)
        self.texts.append(itr_str)

    def set_text_fig(self, text:str, coord_text:tuple[float, float]=(0.4, 0.9), to_update=False, *args, **kwargs):
        """Пользовательский текст для окна, to_update=True позволяет далее обновлять его"""
        text_obj = self.fig.text(*coord_text, text, *args, **kwargs)
        if to_update:
            self.texts.append(text_obj)
        self.fig.subplots_adjust(top=coord_text[1] - 0.02, bottom=0.1)

    def set_text_axis(self, text:str, ax_idx:int = 0, coord_text:tuple[float, float]=(0.1, 0.95), to_update=False, **kwargs):
        """Пользовательский текст для оси по индексу ax_idx (если осей > 1), to_update=True позволяет далее обновлять его"""
        selected_axis = self._select_axis(ax_idx)

        settings = {'transform': selected_axis.transAxes,
                      'fontsize' : 12,
                      'verticalalignment' : 'top',
                      'bbox' : dict(  # 6. Подложка (рамка)
                            boxstyle='round',  # — закруглённые углы
                            facecolor='white',  # — цвет фона
                            alpha=0.8  # — прозрачность (0.8 = почти непрозрачный)
                      )
            }
        for k, v in kwargs.items():
            settings[k] = v

        text_obj = selected_axis.text(*coord_text, text, **settings)
        if to_update:
            self.texts.append(text_obj)

    def update_text(self, idx:int, new_text:str):
        """Обновить текст по индексу в списке"""
        self.texts[idx].set_text(new_text)

    def draw_graph(self, x_data, y_data, ax_idx:int = 0, axis_name=('x', 'y'), **kwargs):
        """Делает график на оси по индексу ax_idx (если осей > 1)"""
        selected_axis = self._select_axis(ax_idx)

        settings = dict(color='blue', linewidth=3, label=None)
        for k, v in kwargs.items():
            settings[k]= v

        graph_obj = selected_axis.plot(x_data, y_data, **settings)
        selected_axis.set_xlabel(axis_name[0])
        selected_axis.set_ylabel(axis_name[1])
        selected_axis.grid(True, alpha=0.3)
        selected_axis.legend()
        return graph_obj

    def _select_axis(self, ax_idx:int = 0):
        """Принимает индекс, возвращает ось"""
        try:
            selected_axis = self.axes[ax_idx]
        except IndexError:
            raise IndexError(f"Оси с координатами {ax_idx} не существует")
        return selected_axis


class MovePoint(Init):
    def __init__(self, title:str, axis_place:tuple[int, int]=(), *args, **kwargs):
        super().__init__(title, axis_place, *args, **kwargs)

        self.points = []

    def make_point(self, x_data, y_data, ax_idx:int = 0, to_update=False, *args, **kwargs):
        """Делает точку на оси по индексу ax_idx (если осей > 1)"""
        selected_axis = self._select_axis(ax_idx)

        settings_pos = ('ro', )
        settings_names = dict(markersize=12)
        if args:
            settings_pos = args
        for k, v in kwargs.items():
            settings_names[k] = v

        point = selected_axis.plot([x_data], [y_data], *settings_pos, **settings_names)
        if to_update:
            self.points.append(*point)

    def update_point(self, new_x, new_y, idx:int):
        """Обновляет точку по индексу idx в списке"""
        self.points[idx].set_data([new_x], [new_y])


class DynamicGraphs(Init):
    def __init__(self, title:str, axis_place:tuple[int, int]=(), *args, **kwargs):
        super().__init__(title, axis_place, *args, **kwargs)
        
        self.graphs = []

    def draw_graph(self, x_data, y_data, ax_idx: int = 0, to_update=False, axis_name=('x', 'y'), **custom_settings):
        graph = super().draw_graph(x_data, y_data, ax_idx, axis_name, **custom_settings)
        if to_update:
            self.graphs.append(*graph)

    def update_graphs(self, x_data, y_data, idx:int=0, new_label=None):
        graph_obj = self.graphs[idx]
        graph_obj.set_data(x_data, y_data)
        if new_label:
            graph_obj.set_label(new_label)

    def update_legend(self, ax_idx: int = 0, loc: str = 'lower right'):
        ax = self.axes[ax_idx]
        ax.legend(loc=loc)


class ClassificationPlot(Init):
    def __init__(self, title:str, axis_place:tuple[int, int]=(), *args, **kwargs):
        super().__init__(title, axis_place, *args, **kwargs)

        self.graphs = []
        self.lim_x = None

    def __set_lims_ax(self, coords, ax_idx: int = 0, ε:float=0.5):
        x_min = min(coords, key=lambda t: t[0])[0] - ε
        x_max = max(coords, key=lambda t: t[0])[0] + ε
        self.lim_x = x_min, x_max
        ax = self._select_axis(ax_idx)
        ax.set_xlim(x_min, x_max)

        y_min = min(coords, key=lambda t: t[1])[1] - ε
        y_max = max(coords, key=lambda t: t[1])[1] + ε
        ax.set_ylim(y_min, y_max)

    def draw_cls_points(self, coords, classes, ax_idx: int = 0, ε:float=0.5):
        self.__set_lims_ax(coords, ax_idx, ε)
        classes = np.array(classes)
        coords = np.array(coords)
        ax = self._select_axis(ax_idx)
        colors = (
            'blue', 'red', 'green', 'orange', 'purple',
            'brown', 'pink', 'gray', 'cyan', 'magenta',
            'olive', 'teal', 'navy', 'coral', 'lime'
        )
        classes_set = tuple(set(classes))
        if len(classes_set) > len(colors):
            raise ValueError("Слишком много классов")

        for cls, color in zip(classes_set, colors):
            x = coords[classes == cls]
            ax.scatter(x[:, 0], x[:, 1], color=color)

    def draw_line_2D(self, coords:list[tuple, tuple]=None, ax_idx:int = 0, to_update=False, **custom_settings):
        if coords:
            x = coords[0][0], coords[1][0]
            y = coords[0][1], coords[1][1]
        else:
            x = self.lim_x
            y = (0, 0)

        graph_obj = super().draw_graph(x, y, ax_idx, **custom_settings)
        if to_update:
            self.graphs.append(*graph_obj)

    def update_line_2D(self, w:np.ndarray, idx:int=0):
        line_obj = self.graphs[idx]
        x = self.lim_x

        w_norm = [w[1] / w[2], w[0] / w[2]]
        updated_y = [-x_i * w_norm[0] - w_norm[1] for x_i in x]
        line_obj.set_data(x, updated_y)