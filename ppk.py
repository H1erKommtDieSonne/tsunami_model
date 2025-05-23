import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

reliefs = []
sources = []

def add_relief():
    rtype = relief_type_var.get()
    x = int(relief_x_entry.get())
    y = int(relief_y_entry.get())
    h = float(relief_h_entry.get())
    r = float(relief_r_entry.get())
    reliefs.append({'type': rtype, 'x': x, 'y': y, 'h': h, 'r': r})
    refresh_relief_list()

def add_source():
    x = int(source_x_entry.get())
    y = int(source_y_entry.get())
    a = float(source_a_entry.get())
    r = float(source_r_entry.get())
    f = float(source_f_entry.get())
    sources.append({'x': x, 'y': y, 'a': a, 'r': r, 'f': f})
    refresh_source_list()

def delete_selected_relief():
    selected = relief_listbox.curselection()
    if selected:
        del reliefs[selected[0]]
        refresh_relief_list()

def delete_selected_source():
    selected = source_listbox.curselection()
    if selected:
        del sources[selected[0]]
        refresh_source_list()

def refresh_relief_list():
    relief_listbox.delete(0, tk.END)
    for i, r in enumerate(reliefs):
        relief_listbox.insert(tk.END, f"{i+1}: {r['type']} (x={r['x']}, y={r['y']}, h={r['h']}, r={r['r']})")

def refresh_source_list():
    source_listbox.delete(0, tk.END)
    for i, s in enumerate(sources):
        source_listbox.insert(tk.END, f"{i+1}: Источник (x={s['x']}, y={s['y']}, A={s['a']}, R={s['r']}, f={s['f']})")

def generate_field():
    Nx, Ny = 100, 100
    Lx, Ly = 100.0, 100.0
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    g = 9.81
    D = np.ones((Ny, Nx)) * 10

    for r in reliefs:
        fx = np.exp(-((X - r['x'])**2 + (Y - r['y'])**2) / (2 * r['r']**2))
        if r['type'] == 'Гора':
            D -= r['h'] * fx
        elif r['type'] == 'Впадина':
            D += r['h'] * fx
        elif r['type'] == 'Хребет':
            D -= r['h'] * np.exp(-((X - r['x'])**2) / (2 * r['r']**2))
        elif r['type'] == 'Плато':
            D += r['h'] * (fx > np.exp(-0.5))

    D = np.clip(D, 1, None)
    c = np.sqrt(g * D)

    eta = np.zeros((Ny, Nx))
    for s in sources:
        for i in range(Ny):
            for j in range(Nx):
                r2 = (i - s['y'])**2 + (j - s['x'])**2
                eta[i, j] += s['a'] * np.exp(-r2 / (2 * s['r']**2))

    eta_old = eta.copy()
    dt = 0.05
    T = 20
    Nt = int(T / dt)
    return X, Y, eta, eta_old, c, dt, Nt

def run_simulation_2d():
    try:
        X, Y, eta, eta_old, c, dt, Nt = generate_field()
        Ny, Nx = eta.shape
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]

        fig, ax = plt.subplots()
        img = ax.imshow(eta, extent=[0, 100, 0, 100], origin='lower', cmap='viridis', vmin=-1, vmax=1)
        fig.colorbar(img, ax=ax, label="Высота η(x, y)")
        ax.set_title("2D-анимация волн")

        def update(frame):
            nonlocal eta, eta_old
            eta_new = np.zeros_like(eta)
            for i in range(1, Ny - 1):
                for j in range(1, Nx - 1):
                    d2x = (eta[i, j+1] - 2*eta[i, j] + eta[i, j-1]) / dx**2
                    d2y = (eta[i+1, j] - 2*eta[i, j] + eta[i-1, j]) / dy**2
                    eta_new[i, j] = (2 * eta[i, j] - eta_old[i, j] + (dt**2) * c[i, j]**2 * (d2x + d2y))
            for s in sources:
                if s['f'] > 0 and frame % int(1/(s['f'] * dt)) == 0:
                    for i in range(Ny):
                        for j in range(Nx):
                            r2 = (i - s['y'])**2 + (j - s['x'])**2
                            eta_new[i, j] += s['a'] * np.exp(-r2 / (2 * s['r']**2))
            eta_new[0, :] = eta_new[1, :]
            eta_new[-1, :] = eta_new[-2, :]
            eta_new[:, 0] = eta_new[:, 1]
            eta_new[:, -1] = eta_new[:, -2]
            eta_old, eta = eta, eta_new
            img.set_array(eta)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=Nt, interval=30, blit=True)
        plt.show()

    except Exception as e:
        messagebox.showerror("Пупупу, непорядок", str(e))

def run_simulation_3d():
    try:
        X, Y, eta, eta_old, c, dt, Nt = generate_field()
        Ny, Nx = eta.shape
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(-1, 1)
        surf = [ax.plot_surface(X, Y, eta, cmap='viridis', vmin=-1, vmax=1)]

        def update(frame):
            nonlocal eta, eta_old
            eta_new = np.zeros_like(eta)
            for i in range(1, Ny - 1):
                for j in range(1, Nx - 1):
                    d2x = (eta[i, j+1] - 2*eta[i, j] + eta[i, j-1]) / dx**2
                    d2y = (eta[i+1, j] - 2*eta[i, j] + eta[i-1, j]) / dy**2
                    eta_new[i, j] = (2 * eta[i, j] - eta_old[i, j] + (dt**2) * c[i, j]**2 * (d2x + d2y))
            for s in sources:
                if s['f'] > 0 and frame % int(1/(s['f'] * dt)) == 0:
                    for i in range(Ny):
                        for j in range(Nx):
                            r2 = (i - s['y'])**2 + (j - s['x'])**2
                            eta_new[i, j] += s['a'] * np.exp(-r2 / (2 * s['r']**2))
            eta_new[0, :] = eta_new[1, :]
            eta_new[-1, :] = eta_new[-2, :]
            eta_new[:, 0] = eta_new[:, 1]
            eta_new[:, -1] = eta_new[:, -2]
            eta_old, eta = eta, eta_new
            ax.clear()
            ax.set_zlim(-1, 1)
            ax.plot_surface(X, Y, eta, cmap='viridis', vmin=-1, vmax=1)
            ax.set_title("3D-анимация волн")
            return []

        ani = animation.FuncAnimation(fig, update, frames=Nt, interval=30, blit=False)
        plt.show()

    except Exception as e:
        messagebox.showerror("пупупу, вот неприятность то", str(e))

# --- Интерфейс ---
root = tk.Tk()
root.title("Модель цунами 2D / 3D")

ttk.Label(root, text="Тип рельефа:").grid(row=0, column=0)
relief_type_var = tk.StringVar(value="Гора")
ttk.Combobox(root, textvariable=relief_type_var, values=["Гора", "Впадина", "Хребет", "Плато"]).grid(row=0, column=1)

ttk.Label(root, text="Центр (x, y):").grid(row=1, column=0)
relief_x_entry = ttk.Entry(root); relief_x_entry.insert(0, "50"); relief_x_entry.grid(row=1, column=1)
relief_y_entry = ttk.Entry(root); relief_y_entry.insert(0, "50"); relief_y_entry.grid(row=1, column=2)

ttk.Label(root, text="Глубина/высота:").grid(row=2, column=0)
relief_h_entry = ttk.Entry(root); relief_h_entry.insert(0, "5"); relief_h_entry.grid(row=2, column=1)

ttk.Label(root, text="Радиус:").grid(row=3, column=0)
relief_r_entry = ttk.Entry(root); relief_r_entry.insert(0, "8"); relief_r_entry.grid(row=3, column=1)

ttk.Button(root, text="Добавить рельеф", command=add_relief).grid(row=4, column=0)
ttk.Button(root, text="Удалить рельеф", command=delete_selected_relief).grid(row=4, column=1)
relief_listbox = tk.Listbox(root, height=4); relief_listbox.grid(row=5, column=0, columnspan=3)

ttk.Label(root, text="Источник (x, y):").grid(row=6, column=0)
source_x_entry = ttk.Entry(root); source_x_entry.insert(0, "25"); source_x_entry.grid(row=6, column=1)
source_y_entry = ttk.Entry(root); source_y_entry.insert(0, "25"); source_y_entry.grid(row=6, column=2)

ttk.Label(root, text="Амплитуда:").grid(row=7, column=0)
source_a_entry = ttk.Entry(root); source_a_entry.insert(0, "3"); source_a_entry.grid(row=7, column=1)

ttk.Label(root, text="Радиус:").grid(row=8, column=0)
source_r_entry = ttk.Entry(root); source_r_entry.insert(0, "5"); source_r_entry.grid(row=8, column=1)

ttk.Label(root, text="Частота (0 — один раз):").grid(row=9, column=0)
source_f_entry = ttk.Entry(root); source_f_entry.insert(0, "0"); source_f_entry.grid(row=9, column=1)

ttk.Button(root, text="Добавить источник", command=add_source).grid(row=10, column=0)
ttk.Button(root, text="Удалить источник", command=delete_selected_source).grid(row=10, column=1)
source_listbox = tk.Listbox(root, height=4); source_listbox.grid(row=11, column=0, columnspan=3)

ttk.Button(root, text="Запустить 2D-анимацию", command=run_simulation_2d).grid(row=12, column=0, columnspan=3, pady=5)
ttk.Button(root, text="Запустить 3D-анимацию", command=run_simulation_3d).grid(row=13, column=0, columnspan=3, pady=5)

root.mainloop()
