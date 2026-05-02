import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import csv

from ga_core import RealGeneticAlgorithm
from functions import AVAILABLE_FUNCTIONS

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algorytm genetyczny — chromosom rzeczywisty (P2)")
        self.geometry("1100x700")
        
        self._build_ui()
        
    def _build_ui(self):
        control_frame = ttk.LabelFrame(self, text="Konfiguracja parametrów")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        row = 0
        def add_entry(label, default):
            nonlocal row
            ttk.Label(control_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            entries[label] = ttk.Entry(control_frame, width=15)
            entries[label].insert(0, default)
            entries[label].grid(row=row, column=1, pady=2, padx=5)
            row += 1

        def add_combo(label, values, default):
            nonlocal row
            ttk.Label(control_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            combos[label] = ttk.Combobox(control_frame, values=values, state="readonly", width=13)
            combos[label].set(default)
            combos[label].grid(row=row, column=1, pady=2, padx=5)
            row += 1

        entries = {}
        combos = {}
        
        add_combo("Funkcja", list(AVAILABLE_FUNCTIONS.keys()), list(AVAILABLE_FUNCTIONS.keys())[0])
        add_combo("Rodzaj optymalizacji", ["Min", "Max"], "Min")
        add_entry("Liczba zmiennych", "2")
        add_entry("Min", "-10.0")
        add_entry("Max", "10.0")
        add_entry("Wyświetlanie miejsc po przecinku", "5")
        add_entry("Populacja", "100")
        add_entry("Liczba epok", "200")
        add_combo("Metoda selekcji", ["Najlepszych", "Turniejowa", "Ruletki"], "Turniejowa")
        add_entry("Rozmiar turnieju / % najlepszych", "3")
        add_combo(
            "Rodzaj krzyżowania",
            ["Arytmetyczne", "Liniowe", "Mieszające alfa", "Alfa-beta", "Uśredniające"],
            "Arytmetyczne",
        )
        add_entry("Prawdopodobieństwo krzyżowania", "0.8")
        add_combo("Rodzaj mutacji", ["Równomierna", "Gaussa"], "Równomierna")
        add_entry("Sigma mutacji Gaussa (ułamek szerokości dziedziny)", "0.1")
        add_entry("Prawdopodobieństwo mutacji", "0.05")
        add_entry("Prawdopodobieństwo inwersji", "0.02")
        add_entry("Liczba elit", "1")
        
        self.entries = entries
        self.combos = combos
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        self.btn_run = ttk.Button(btn_frame, text="Uruchom", command=self.run_algorithm)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        
        self.btn_export = ttk.Button(btn_frame, text="Eksportuj do CSV", command=self.export_csv, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5)

        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.res_lbl = ttk.Label(right_frame, text="", font=("Helvetica", 11))
        self.res_lbl.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
    def get_val(self, label, type_fn=float):
        try:
            return type_fn(self.entries[label].get())
        except ValueError:
             raise ValueError(f"Nieprawidłowa wartość dla '{label}'")
            
    def run_algorithm(self):
        try:
            func_name = self.combos["Funkcja"].get()
            opt_type = self.combos["Rodzaj optymalizacji"].get()
            num_vars = self.get_val("Liczba zmiennych", int)
            domain_min = self.get_val("Min", float)
            domain_max = self.get_val("Max", float)
            display_decimals = self.get_val("Wyświetlanie miejsc po przecinku", int)
            pop_size = self.get_val("Populacja", int)
            epochs = self.get_val("Liczba epok", int)
            
            sel_method = self.combos["Metoda selekcji"].get()
            tour_size_or_pct = self.get_val("Rozmiar turnieju / % najlepszych", float)
            
            cross_method = self.combos["Rodzaj krzyżowania"].get()
            cross_prob = self.get_val("Prawdopodobieństwo krzyżowania", float)
            
            mut_method = self.combos["Rodzaj mutacji"].get()
            gauss_sigma_frac = self.get_val("Sigma mutacji Gaussa (ułamek szerokości dziedziny)", float)
            mut_prob = self.get_val("Prawdopodobieństwo mutacji", float)
            
            inv_prob = self.get_val("Prawdopodobieństwo inwersji", float)
            elitism = self.get_val("Liczba elit", int)
            
        except ValueError as e:
            messagebox.showerror("Błąd konfiguracji", str(e))
            return
            
        fitness_fn = AVAILABLE_FUNCTIONS[func_name]
        
        def best_sel_from_entry(v):
            if sel_method != "Najlepszych":
                return 0.2
            if v > 1.0:
                return min(1.0, v / 100.0)
            return min(1.0, max(0.01, v))

        try:
            ga = RealGeneticAlgorithm(
                fitness_func=fitness_fn,
                num_vars=num_vars,
                domain=(domain_min, domain_max),
                pop_size=pop_size,
                epochs=epochs,
                prob_cross=cross_prob,
                prob_mut=mut_prob,
                prob_inv=inv_prob,
                elitism_count=elitism,
                selection_method=sel_method,
                crossover_method=cross_method,
                mutation_method=mut_method,
                opt_type=opt_type,
                tournament_size=int(tour_size_or_pct) if sel_method == "Turniejowa" else 3,
                best_sel_percent=best_sel_from_entry(tour_size_or_pct),
                gaussian_sigma_frac=max(1e-9, gauss_sigma_frac),
            )
        except Exception as e:
            messagebox.showerror("Błąd inicjalizacji GA", str(e))
            return
            
        self.btn_run.config(state=tk.DISABLED)
        
        def worker():
            start_time = time.time()
            try:
                ga.run()
                end_time = time.time()
                
                self.last_ga = ga
                
                self.after(0, self.update_results, ga, end_time - start_time, display_decimals)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Błąd", f"Błąd:\n{e}"))
                self.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
                self.after(0, lambda: self.res_lbl.config(text="Błąd"))
            
        threading.Thread(target=worker, daemon=True).start()

    def update_results(self, ga, duration, display_decimals=5):
        best_val = ga.best_solution_ever.objective_val
        d = max(0, min(12, int(display_decimals)))
        best_vars = [round(x, d) for x in ga.best_solution_ever.real_values]

        if len(best_vars) > 10:
            formatted_vars = str(best_vars[:10])[:-1] + ", ...]"
        else:
            formatted_vars = str(best_vars)
            
        res_text = (f"Czas: {duration*1000:.2f} ms\n"
                    f"Najlepsza wartość: {best_val:.6f}\n"
                    f"Parametry w optimum: {formatted_vars}")
        self.res_lbl.config(text=res_text)
        
        self.ax.clear()
        epochs_range = range(len(ga.best_history))
        self.ax.plot(epochs_range, ga.best_history, label="Najlepszy", color="green")
        self.ax.plot(epochs_range, ga.avg_history, label="Średni", color="blue")
        #self.ax.plot(epochs_range, ga.worst_history, label="Najgorszy", color="red")
        
        self.ax.set_title("Wartość funkcji celu w kolejnych epokach")
        self.ax.set_xlabel("Epoka")
        self.ax.set_ylabel("Wartość")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()
        
        self.btn_run.config(state=tk.NORMAL)
        self.btn_export.config(state=tk.NORMAL)
        
    def export_csv(self):
        if not hasattr(self, 'last_ga'):
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Epoka", "Najlepszy", "Sredni", "Najgorszy"])
                for ep, (b, a, w) in enumerate(zip(self.last_ga.best_history, self.last_ga.avg_history, self.last_ga.worst_history)):
                    writer.writerow([ep, b, a, w])
            messagebox.showinfo("Eksport", "Pomyślnie zapisano")

if __name__ == "__main__":
    app = App()
    app.mainloop()
