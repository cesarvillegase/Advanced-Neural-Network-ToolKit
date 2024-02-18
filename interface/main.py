import tkinter
import tkinter.messagebox
from typing import Optional, Tuple, Union
import customtkinter
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Advanced Neural Network ToolKit")
        self.geometry(f"{1100}x{580}")
        
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        # Create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        # Create a logo label inside the sidebar
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="ANN ToolKit", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Create a label and a option menu for the levels of abstraction
        self.levels_of_abstraction_label = customtkinter.CTkLabel(self.sidebar_frame, text="Levels of Abstraction:", anchor="w")
        self.levels_of_abstraction_label.grid(row=3, column=0, padx=20, pady=(10,0))
        self.levels_of_abstraction_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Low Level", "High Level"])
                                                                        #, command=self.change_level_ofabstraction
        self.levels_of_abstraction_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 20))
        
        # Create a label and a button to open a file
        self.open_file_label = customtkinter.CTkLabel(self.sidebar_frame, text="Open a file:", anchor="w")
        self.open_file_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.open_file_button = customtkinter.CTkButton(self.sidebar_frame, text="Open file", bg_color="red", command=self.sidebar_open_button_event, anchor="w")
        self.open_file_button.grid(row=6, column=0, padx=20, pady=(10, 20))
        
        # Create the label for the author and the description
        self.author_label = customtkinter.CTkLabel(self.sidebar_frame, text="Created by:", font=customtkinter.CTkFont(size=14, weight="normal"))
        self.author_label.grid(row=7, column=0, padx=20, pady=(10,0))
        self.author1_label = customtkinter.CTkLabel(self.sidebar_frame, text="Cesar A Villegas Espindola", font=customtkinter.CTkFont(size=14, weight="normal"))
        self.author1_label.grid(row=8, column=0, padx=20, pady=(10,20))

        # Create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.tabview.add("Hopfield")
        self.tabview.add("Backpropagation")
        self.tabview.add("Kohonen SOM")
        self.tabview.add("Autoencoder")
        self.tabview.add("LVQ")
        
            
    def sidebar_open_button_event(self):
        filetypes = {
            ('csv files', '*.csv'),
            ('All files', '*.')
        }
        
        filename =fd.askopenfilename(
            title ='Open a file',
            initialdir='/',
            filetypes=filetypes)
        
        showinfo(
            title='Selected File',
            message=filename
        )
    

if __name__ == "__main__":
    app = App()
    app.mainloop()