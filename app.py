import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import csv

from ga_core import GeneticAlgorithm
from functions import AVAILABLE_FUNCTIONS

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algorytm Genetyczny - Projekt 1")
        self.geometry("1100x700")
        
        self._build_ui()
        
    def _build_ui(self):
        # Left Panel (Controls)
        control_frame = ttk.LabelFrame(self, text="Konfiguracja Parametrów")
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
        
        add_combo("Funkcja Testowa", list(AVAILABLE_FUNCTIONS.keys()), list(AVAILABLE_FUNCTIONS.keys())[0])
        add_combo("Rodzaj Optymalizacji", ["Min", "Max"], "Min")
        add_entry("Liczba Zmiennych", "2")
        add_entry("Dziedzina (min)", "-10.0")
        add_entry("Dziedzina (max)", "10.0")
        add_entry("Dokładność (miejsca po przecinku)", "3")
        add_entry("Wielkość Populacji", "100")
        add_entry("Liczba Epok", "200")
        add_combo("Metoda Selekcji", ["Najlepszych", "Turniejowa", "Ruletki"], "Turniejowa")
        add_entry("Rozmiar Turnieju / % Najlepszych", "3") # Reused depends on selection
        add_combo("Rodzaj Krzyżowania", ["Jednopunktowe", "Dwupunktowe", "Jednorodne", "Ziarniste"], "Jednopunktowe")
        add_entry("Prawd. Krzyżowania (0-1)", "0.8")
        add_combo("Rodzaj Mutacji", ["Brzegowa", "Jednopunktowa", "Dwupunktowa"], "Jednopunktowa")
        add_entry("Prawd. Mutacji (0-1)", "0.05")
        add_entry("Prawd. Inwersji (0-1)", "0.02")
        add_entry("Liczba elit (Elityzm)", "1")
        
        self.entries = entries
        self.combos = combos
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        self.btn_run = ttk.Button(btn_frame, text="Uruchom Algorytm", command=self.run_algorithm)
        self.btn_run.pack(side=tk.LEFT, padx=5)
        
        self.btn_export = ttk.Button(btn_frame, text="Eksportuj do CSV", command=self.export_csv, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5)
        
        # Right Panel (Chart & Results)
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.res_lbl = ttk.Label(right_frame, text="Wyniki będą tutaj...", font=("Helvetica", 11))
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
            func_name = self.combos["Funkcja Testowa"].get()
            opt_type = self.combos["Rodzaj Optymalizacji"].get()
            num_vars = self.get_val("Liczba Zmiennych", int)
            domain_min = self.get_val("Dziedzina (min)", float)
            domain_max = self.get_val("Dziedzina (max)", float)
            precision = self.get_val("Dokładność (miejsca po przecinku)", int)
            pop_size = self.get_val("Wielkość Populacji", int)
            epochs = self.get_val("Liczba Epok", int)
            
            sel_method = self.combos["Metoda Selekcji"].get()
            tour_size_or_pct = self.get_val("Rozmiar Turnieju / % Najlepszych", float)
            
            cross_method = self.combos["Rodzaj Krzyżowania"].get()
            cross_prob = self.get_val("Prawd. Krzyżowania (0-1)", float)
            
            mut_method = self.combos["Rodzaj Mutacji"].get()
            mut_prob = self.get_val("Prawd. Mutacji (0-1)", float)
            
            inv_prob = self.get_val("Prawd. Inwersji (0-1)", float)
            elitism = self.get_val("Liczba elit (Elityzm)", int)
            
        except ValueError as e:
            messagebox.showerror("Błąd konfiguracji", str(e))
            return
            
        fitness_fn = AVAILABLE_FUNCTIONS[func_name]
        
        try:
            ga = GeneticAlgorithm(
                fitness_func=fitness_fn,
                num_vars=num_vars,
                domain=(domain_min, domain_max),
                precision_decimals=precision,
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
                tournament_size=int(tour_size_or_pct) if sel_method=="Turniejowa" else 3,
                best_sel_percent=tour_size_or_pct/100.0 if sel_method=="Najlepszych" and tour_size_or_pct > 1 else (tour_size_or_pct if sel_method=="Najlepszych" else 0.2)
            )
        except Exception as e:
            messagebox.showerror("Błąd inicjalizacji GA", str(e))
            return
            
        self.btn_run.config(state=tk.DISABLED)
        self.res_lbl.config(text="Obliczanie...")
        
        def worker():
            start_time = time.time()
            try:
                ga.run()
                end_time = time.time()
                
                self.last_ga = ga
                
                self.after(0, self.update_results, ga, end_time - start_time)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Błąd", f"Błąd w trakcie wykonywania:\n{e}"))
                self.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
                self.after(0, lambda: self.res_lbl.config(text="Błąd!"))
            
        threading.Thread(target=worker, daemon=True).start()

    def update_results(self, ga, duration):
        best_val = ga.best_solution_ever.objective_val
        best_vars = [round(x, 5) for x in ga.best_solution_ever.real_values]
        
        # Format the parameters string nicely
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
        self.ax.plot(epochs_range, ga.best_history, label="Najlepszy (Best)", color="green")
        self.ax.plot(epochs_range, ga.avg_history, label="Średnio (Average)", color="blue")
        # Można odkomentować aby rysować 'Najgorszy':
        # self.ax.plot(epochs_range, ga.worst_history, label="Najgorszy (Worst)", color="red")
        
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
            messagebox.showinfo("Eksport", "Pomyślnie zapisano do pliku CSV.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
