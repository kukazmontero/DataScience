import tkinter as tk
import subprocess

def run_classificator():
    # Usar la ruta completa al intérprete Python del entorno virtual
    subprocess.run([r'./roseapp/Scripts/python', 'classificator.py'])

window = tk.Tk()
window.title("Rose AI")

# Establecer el tamaño de la ventana a 400x400
window.geometry('400x400')

frame = tk.Frame(window)
frame.place(relx=0.5, rely=0.5, anchor='center')

# Hacer el botón "elegante" cambiando sus colores, fuente, etc.
button = tk.Button(frame, 
                   text="EJECUTAR", 
                   command=run_classificator, 
                   fg="white",
                   bg="dark blue",
                   font=('Helvetica', 16),
                   relief="raised",
                   borderwidth=3)
button.pack()

window.mainloop()
