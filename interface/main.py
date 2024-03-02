import customtkinter
from tkinter import filedialog as fd, StringVar
from tkinter.messagebox import showinfo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

from neural_networks.backprop import Backpropagation, plot_loss

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Advanced Neural Network ToolKit")
        self.geometry(f"{1280}x{640}")

        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.create_sidebar()
        # Create tabview
        self.create_tabview()

    def create_sidebar(self):
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

    def create_tabview(self):
        # Create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.tabview.add("Hopfield")
        self.tabview.set("Hopfield")  # set currently visible tab
        self.tabview.add("Backpropagation")
        self.tabview.add("Kohonen SOM")
        self.tabview.add("AutoEncoder")
        self.tabview.add("LVQ")

        self.create_tab_1()
        self.create_tab_2()
        self.create_tab_3()
        self.create_tab_4()
        self.create_tab_5()

    def create_tab_1(self):
        # Hopfield tab
        tab_1 = self.tabview.tab("Hopfield")

        self.label_tab_1 = customtkinter.CTkLabel(tab_1, text="Hopfield Network")
        self.label_tab_1.grid(row=0, column=0, padx=20, pady=20)

        # To do: Add the area to add images for the input
        # Create a canvas to display the plot in the Hopfield tab
        self.button1_tab_1 = customtkinter.CTkButton(tab_1, fg_color="transparent", border_width=2,
                                                     text="Generate and Show Plot",
                                                     text_color=("gray10", "#DCE4EE"),
                                                     command=self.generate_and_show_plot, anchor="w")
        self.button1_tab_1.grid(row=1, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.canvas_tab_1 = customtkinter.CTkCanvas(tab_1, width=0, height=0)
        self.canvas_tab_1.grid(row=1, column=3, padx=0, pady=0)

    def create_tab_2(self):
        # Backpropagation tab
        tab_2 = self.tabview.tab("Backpropagation")

        self.label_tab_2 = customtkinter.CTkLabel(tab_2, text="Backpropagation Network")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)

        # Instantiate an object of the Backpropagation class
        self.backpropagation_model = Backpropagation(input_neurons=3,hidden_neurons=3, output_neurons=1)

        input_ = StringVar()
        desired_output_ = StringVar()
        learning_rate_ = StringVar()

        def entry_1_tab_2():
            def parse_input_string(input_string):
                # Remove brackets and split the string into indivual elements
                elements = input_string.replace("[", "").replace("]", "").split(",")

                # Convert elements to integers
                try:
                    elements = [int(element) for element in elements]
                except ValueError:
                    return None

                # Determine the sublist length based on the number of elements
                sublist_length = len(elements) // 4  # Assuming there are four sublists in the input

                # Check if the number of elements is divisible by the calculated sublist_length
                if len(elements) % sublist_length != 0:
                    return None

                # Create sublist
                sublists = [elements[i:i + sublist_length] for i in range(0, len(elements), sublist_length)]

                return sublists

            def print_input_list(input_list):
                if input_list is not None:
                    print(input_list)
                else:
                    print("Invalid input format")

            def print_input_as_list(*args):
                input_value_str = input_.get()
                input_list_parsed = parse_input_string(input_value_str)
                print_input_list(input_list_parsed)

            self.entry1_tab_2 = customtkinter.CTkEntry(tab_2, placeholder_text="Input", textvariable=input_)
            self.entry1_tab_2.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            input_.trace("w", print_input_as_list)

        def entry_2_tab_2():
            def parse_input_string(input_string):
                # Remove brackets and split the string into individual elements
                elements = input_string.replace("[", "").replace("]", "").split(",")

                # Convert elements to integers
                try:
                    elements = [int(element) for element in elements]
                except ValueError:
                    return None

                # Check if each element forms a single-element sublist
                sublists = [[element] for element in elements]

                return sublists

            def print_input_list(input_list):
                if input_list is not None:
                    print(input_list)
                else:
                    print("Invalid input format")

            def print_input_as_list(*args):
                input_value_str = desired_output_.get()
                input_list_parsed = parse_input_string(input_value_str)
                print_input_list(input_list_parsed)

            # Use desired_output as the textvariable for the entry widget
            self.entry2_tab_2 = customtkinter.CTkEntry(tab_2, placeholder_text="Desired output",
                                                       textvariable=desired_output_)
            self.entry2_tab_2.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the textvariable
            desired_output_.trace("w", print_input_as_list)

        def entry_3_tab_2():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = learning_rate_.get()
                    # Convert the input value to a float
                    learning_rate_value = float(input_value_str)
                    # Use the learning_rate_value here
                    print("Learning rate:", learning_rate_value)
                except ValueError:
                    print("Invalid learning rate format")

            # Use learning_rate as the textvariable for the entry widget
            entry3_tab_2 = customtkinter.CTkEntry(tab_2, placeholder_text="Learning rate", textvariable=learning_rate_)
            entry3_tab_2.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the textvariable
            learning_rate_.trace("w", parse_entry)

        entry_1_tab_2()
        entry_2_tab_2()
        entry_3_tab_2()

        def train_backprop():
            input_data_str = input_.get()  # This is a string; you'll need to convert it to a numpy array
            desired_output_str = desired_output_.get()  # Also a string to convert
            learning_rate_str = learning_rate_.get()  # And this is a string to convert to float

            # Convert the string representations to the appropriate types
            try:
                input_data = np.array(eval(input_data_str))  # Using eval can be risky; ensure the input is sanitized
                desired_output = np.array(eval(desired_output_str))
                learning_rate = float(learning_rate_str)

                # Now, you can use the data to train your neural network using the Backpropagation object
                self.backpropagation_model.train(input_data, desired_output, learning_rate)

                print("Training phase")
            except Exception as e:
                print(f"An error occurred: {e}")

        self.button1_tab_2 = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                     text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                     command=train_backprop)

        self.button1_tab_2.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        def test_backprop():
            input_data_str = input_.get()  # This is a string; you'll need to convert it to a numpy array
            desired_output_str = desired_output_.get()  # Also a string to convert

            # Convert the string representations to the appropriate types
            try:
                input_data = np.array(eval(input_data_str))  # Using eval can be risky; ensure the input is sanitized
                desired_output = np.array(eval(desired_output_str))

                # Now, you can use the data to train your neural network using the Backpropagation object
                self.backpropagation_model.test(input_data, desired_output)

                print("Test phase")
            except Exception as e:
                print(f"An error occurred: {e}")

        self.button2_tab_2 = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                     text="Test Network", text_color=("gray10", "#DCE4EE"),
                                                     command=test_backprop)
        self.button2_tab_2.grid(row=4, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Plot loss
        def plot_loss_from_test():
            # Plot the loss directly using the loss values obtained during the test
            loss_values = self.backpropagation_model.loss

            # Clear the previous plot, if any
            self.canvas_tab_2.delete("all")

            # Create a new figure for the plot
            fig = plt.Figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
            ax.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Mean Square Error')
            ax.set_title("Training loss")
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()

            # Embed the plot into the Tkinter canvas
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_tab_2)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        self.canvas_tab_2 = customtkinter.CTkCanvas(tab_2, width=400, height=300)
        self.canvas_tab_2.grid(row=1, column=2, padx=(30, 0), pady=(0))

        self.button3_tab_2 = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                     text="Plot loss", text_color=("gray10", "#DCE4EE"),
                                                     command=plot_loss_from_test)
        self.button3_tab_2.grid(row=4, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def create_tab_3(self):
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

    def create_tab_4(self):
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

    def create_tab_5(self):
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

    def generate_and_show_plot(self):
        # Example plot generation using Matplotlib
        x = [1, 2, 3, 4, 5]
        y = [2, 3, 5, 7, 6]

        # Clear the previous plot if it exists
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.line, = self.ax.plot(x, y)
        self.ax.set_xlabel('X-axis Label')
        self.ax.set_ylabel('Y-axis Label')
        self.ax.set_title('Sample Plot')

        # Embed the Matplotlib plot into the canvas of the Hopfield tab
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_tab_1)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

        # Explicitly call plt.show() after updating the canvas
        plt.show()


if __name__ == "__main__":
    app = App()
    app.mainloop()
