from matplotlib import pyplot as plt
import numpy as np

def create_window(*args, figsize=(10, 6), title=None, ox_name='x', oy_name='y') -> tuple:
    """Создаёт окно с 2D осями"""
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



def drow2Dgraph(x, y, axis,
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
    drow2Dgraph(x, y, ax, name_func=func_str)
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
