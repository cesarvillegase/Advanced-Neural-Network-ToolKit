import customtkinter
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

customtkinter.set_appearance_mode("system")
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
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="ANN ToolKit",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Create a label and an option menu for the levels of abstraction
        self.levels_of_abstraction_label = customtkinter.CTkLabel(self.sidebar_frame, text="Levels of Abstraction:",
                                                                  anchor="w")
        self.levels_of_abstraction_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.levels_of_abstraction_option_menu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                             values=["Low Level", "High Level"])
        # , command=self.change_level_of_abstraction
        self.levels_of_abstraction_option_menu.grid(row=3, column=0, padx=20, pady=(10, 20))

        # Create a label and a button to open a file
        self.open_file_label = customtkinter.CTkLabel(self.sidebar_frame, text="Open a file:", anchor="w")
        self.open_file_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.open_file_button = customtkinter.CTkButton(self.sidebar_frame, text="Open file",
                                                        command=self.sidebar_open_button_event, anchor="w")
        self.open_file_button.grid(row=5, column=0, padx=20, pady=(10, 20))

        # Create the label for the author and the description
        self.author_label = customtkinter.CTkLabel(self.sidebar_frame, text="Created by:",
                                                   font=customtkinter.CTkFont(size=14, weight="normal"))
        self.author_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.author1_label = customtkinter.CTkLabel(self.sidebar_frame, text="Cesar A Villegas Espindola",
                                                    font=customtkinter.CTkFont(size=14, weight="normal"))
        self.author1_label.grid(row=7, column=0, padx=20, pady=(10, 20))

        self.exit_button = customtkinter.CTkButton(self.sidebar_frame, fg_color="red", text="Close App",
                                                   command=self.exit, anchor="w")
        self.exit_button.grid(row=8, column=0, padx=20, pady=(10, 20))

        # Create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.tabview.add("Hopfield")
        self.tabview.set("Hopfield")  # set currently visible tab
        self.tabview.add("Backpropagation")
        self.tabview.add("Kohonen SOM")
        self.tabview.add("AutoEncoder")
        self.tabview.add("LVQ")

        # Hopfield tab
        tab_1 = self.tabview.tab("Hopfield")

        # Backpropagation tab
        tab_2 = self.tabview.tab("Backpropagation")

        self.label_tab_2 = customtkinter.CTkLabel(tab_2, text="Backpropagation Network")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)

        self.entry1_tab_2 = customtkinter.CTkEntry(tab_2, placeholder_text="Input")
        self.entry1_tab_2.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry2_tab_2 = customtkinter.CTkEntry(tab_2, placeholder_text="Desired output")
        self.entry2_tab_2.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry3_tab_2 = customtkinter.CTkEntry(tab_2, placeholder_text="Learning rate")
        self.entry3_tab_2.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.button1_tab_2 = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                     text="Train Network", text_color=("gray10", "#DCE4EE"))
        self.button1_tab_2.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button2_tab_2 = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                     text="Test Network", text_color=("gray10", "#DCE4EE"))
        self.button2_tab_2.grid(row=4, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button3_tab_2 = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                     text="Plot loss", text_color=("gray10", "#DCE4EE"))
        self.button3_tab_2.grid(row=4, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Kohonen Som tab
        tab_3 = self.tabview.tab("Kohonen SOM")

        self.label_tab_3 = customtkinter.CTkLabel(tab_3, text="Kohonen SOM Network")
        self.label_tab_3.grid(row=0, column=0, padx=20, pady=20)

        self.entry1_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Input")
        self.entry1_tab_3.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry2_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Input Dimension")
        self.entry2_tab_3.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry3_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Num of neurons")
        self.entry3_tab_3.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry4_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Learning rate")
        self.entry4_tab_3.grid(row=4, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry5_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Epoch max")
        self.entry5_tab_3.grid(row=5, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.button1_tab_3 = customtkinter.CTkButton(tab_3, fg_color="transparent", border_width=2,
                                                     text="Train Network", text_color=("gray10", "#DCE4EE"))
        self.button1_tab_3.grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button2_tab_3 = customtkinter.CTkButton(tab_3, fg_color="transparent", border_width=2,
                                                     text="Test Network", text_color=("gray10", "#DCE4EE"))
        self.button2_tab_3.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Autoencoder tab
        tab_4 = self.tabview.tab("AutoEncoder")

        self.label_tab_4 = customtkinter.CTkLabel(tab_4, text="AutoEncoder Network")
        self.label_tab_4.grid(row=0, column=0, padx=20, pady=20)

        self.entry1_tab_4 = customtkinter.CTkEntry(tab_4, placeholder_text="Input")
        self.entry1_tab_4.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry2_tab_4 = customtkinter.CTkEntry(tab_4, placeholder_text="Learning rate")
        self.entry2_tab_4.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry3_tab_4 = customtkinter.CTkEntry(tab_4, placeholder_text="Momentum rate")
        self.entry3_tab_4.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry4_tab_4 = customtkinter.CTkEntry(tab_4, placeholder_text="Epoch max")
        self.entry4_tab_4.grid(row=4, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.button1_tab_4 = customtkinter.CTkButton(tab_4, fg_color="transparent", border_width=2,
                                                     text="Train Network", text_color=("gray10", "#DCE4EE"))
        self.button1_tab_4.grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button2_tab_4 = customtkinter.CTkButton(tab_4, fg_color="transparent", border_width=2,
                                                     text="Plot loss", text_color=("gray10", "#DCE4EE"))
        self.button2_tab_4.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # LVQ tab
        tab_5 = self.tabview.tab("LVQ")

        self.label_tab_5 = customtkinter.CTkLabel(tab_5, text="LVQ Network")
        self.label_tab_5.grid(row=0, column=0, padx=20, pady=20)

        self.entry1_tab_5 = customtkinter.CTkEntry(tab_5, placeholder_text="Input")
        self.entry1_tab_5.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry2_tab_5 = customtkinter.CTkEntry(tab_5, placeholder_text="Labels")
        self.entry2_tab_5.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry3_tab_5 = customtkinter.CTkEntry(tab_5, placeholder_text="Learning rate")
        self.entry3_tab_5.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.entry4_tab_5 = customtkinter.CTkEntry(tab_5, placeholder_text="Test input")
        self.entry4_tab_5.grid(row=4, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.button1_tab_5 = customtkinter.CTkButton(tab_5, fg_color="transparent", border_width=2,
                                                     text="Train Network", text_color=("gray10", "#DCE4EE"))
        self.button1_tab_5.grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button2_tab_5 = customtkinter.CTkButton(tab_5, fg_color="transparent", border_width=2,
                                                     text="Test Network", text_color=("gray10", "#DCE4EE"))
        self.button2_tab_5.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

    @staticmethod
    def sidebar_open_button_event():
        filetypes = {
            ('csv files', '*.csv'),
            ('All files', '*.')
        }

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        showinfo(
            title='Selected File',
            message=filename
        )

    def exit(self):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
