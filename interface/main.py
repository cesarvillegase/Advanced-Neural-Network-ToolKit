from tkinter import filedialog as fd, StringVar
from tkinter.messagebox import showinfo

import customtkinter
import matplotlib.pyplot as plt
import numpy as np
from customtkinter import CTkRadioButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# from neural_networks.hopfield import Hopfield
from neural_networks.backprop import Backpropagation
from neural_networks.hopfield import Hopfield
from neural_networks.som_kohonen import SOM
from neural_networks.autoencoder import AutoEncoder

customtkinter.set_appearance_mode("dark")
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
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Create a logo label inside the sidebar
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="ANN ToolKit",
                                            font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Label for the levels of abstraction
        self.levels_of_abstraction_label = customtkinter.CTkLabel(self.sidebar_frame, text="Levels:",
                                                             anchor="w")
        self.levels_of_abstraction_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")

        # Variable to store the level of abstraction choice
        self.abstraction_level_var = customtkinter.StringVar(
            value="Low Level")  # Default value can be "Low Level" or "High Level"

        # Radio button for Low Level
        self.low_level_radio = CTkRadioButton(self.sidebar_frame, text="Low-Level",
                                         variable=self.abstraction_level_var, value="Low Level")
        self.low_level_radio.grid(row=2, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        # Radio button for High Level
        self.high_level_radio = CTkRadioButton(self.sidebar_frame, text="High-Level",
                                          variable=self.abstraction_level_var, value="High Level")
        self.high_level_radio.grid(row=3, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        # Create the label for the author and the description
        self.author_label = customtkinter.CTkLabel(self.sidebar_frame, text="Created by:",
                                              font=customtkinter.CTkFont(size=14, weight="normal"))
        self.author_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.author1_label = customtkinter.CTkLabel(self.sidebar_frame, text="Cesar A Villegas Espindola",
                                               font=customtkinter.CTkFont(size=14, weight="normal"))
        self.author1_label.grid(row=5, column=0, padx=20, pady=(10, 20))

        self.exit_button = customtkinter.CTkButton(self.sidebar_frame, fg_color="red", text="Close App",
                                              command=self.exit, anchor="w")
        self.exit_button.grid(row=6, column=0, padx=20, pady=(10, 20))


        # Create tabview
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.tabview.add("Hopfield")
        self.tabview.set("Hopfield")  # set currently visible tab
        self.tabview.add("Backpropagation")
        self.tabview.add("Kohonen SOM")
        self.tabview.add("AutoEncoder")
        self.tabview.add("LVQ")

        # ########### 1st Tab ###########
        tab_1 = self.tabview.tab("Hopfield")
        # ###############################

        self.label_hop = customtkinter.CTkLabel(tab_1, text="Hopfield Network",
                                                font=("bold", 24))
        self.label_hop.grid(row=0, column=0, padx=20, pady=20)

        epoch_max_hop = StringVar()
        num_og_images = StringVar()
        num_noisy_images = StringVar()

        self.label_epoch_max_hop = customtkinter.CTkLabel(tab_1, text="Epoch max",
                                                          font=("bold", 16))
        self.label_epoch_max_hop.grid(row=1, column=0, padx=20, pady=20)

        def entry_epoch_max_hop():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = epoch_max_hop.get()
                    # Convert the input value to a float
                    epoch_max_hop_value = int(input_value_str)
                    # Use the learning_rate_value here
                    print("Epoch max:", epoch_max_hop_value)
                except ValueError:
                    print("Invalid Epoch max format")

            # Use learning_rate as the text variable for the entry widget
            self.entry_epoch_max_hop = customtkinter.CTkEntry(tab_1, placeholder_text="Epoch max",
                                                              textvariable=epoch_max_hop)
            self.entry_epoch_max_hop.grid(row=1, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            epoch_max_hop.trace("w", parse_entry)

        entry_epoch_max_hop()

        self.hopfield_model = Hopfield(input, epoch_max=epoch_max_hop)

        def entry_num_og_images():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = num_og_images.get()
                    # Convert the input value to a float
                    num_og_images_value = int(input_value_str)
                    # Use the learning_rate_value here
                    print("Number of original images:", num_og_images_value)
                except ValueError:
                    print("Invalid Number of original images format")

            # Use learning_rate as the text variable for the entry widget
            self.entry_num_og_images = customtkinter.CTkEntry(tab_1, placeholder_text="Num of original images",
                                                              textvariable=num_og_images)
            self.entry_num_og_images.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            num_og_images.trace("w", parse_entry)

        def entry_num_noisy_images():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = num_noisy_images.get()
                    # Convert the input value to an integer
                    num_noisy_images_value = int(input_value_str)
                    # Use the num_noisy_images_value here
                    print("Number of noisy images:", num_noisy_images_value)
                except ValueError:
                    print("Invalid Number of noisy images format")

            # Use learning_rate as the text variable for the entry widget
            self.entry_num_noisy_images = customtkinter.CTkEntry(tab_1, placeholder_text="Num of noisy images",
                                                                 textvariable=num_noisy_images)
            self.entry_num_noisy_images.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            num_noisy_images.trace("w", parse_entry)

        entry_num_og_images()
        entry_num_noisy_images()

        # select the image that you want to decode
        # Variable to store the image of choice
        self.image_choice_var = customtkinter.StringVar(value="First Image")  # Default is "First Image"

        # Radio button for the First Image
        self.first_image_radio = customtkinter.CTkRadioButton(tab_1, text="First Image",
                                                              variable=self.image_choice_var, value="First Image")
        self.first_image_radio.grid(row=4, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        # Radio button for the Second Image
        self.second_image_radio = customtkinter.CTkRadioButton(tab_1, text="Second Image",
                                                               variable=self.image_choice_var, value="Second Image")
        self.second_image_radio.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        # Radio button for the Third Image
        self.third_image_radio = customtkinter.CTkRadioButton(tab_1, text="Third Image",
                                                              variable=self.image_choice_var, value="Third Image")
        self.third_image_radio.grid(row=4, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        # To do: Add the area to add images for the input
        # Create a canvas to display the plot in the Hopfield tab
        self.button_train_hop = customtkinter.CTkButton(tab_1, fg_color="transparent", border_width=2,
                                                        text="Train algorithm",
                                                        text_color=("gray10", "#DCE4EE"),
                                                        anchor="w")  # , command=self.generate_and_show_plot
        self.button_train_hop.grid(row=5, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button_reconstruction_hop = customtkinter.CTkButton(tab_1, fg_color="transparent", border_width=2,
                                                                 text="Recontruct image",
                                                                 text_color=("gray10", "#DCE4EE"),
                                                                 anchor="w")
        self.button_reconstruction_hop.grid(row=5, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.canvas_tab_1 = customtkinter.CTkCanvas(tab_1, width=0, height=0)
        self.canvas_tab_1.grid(row=3, column=2, padx=0, pady=0)

        # ########### 2nd Tab ###########
        tab_2 = self.tabview.tab("Backpropagation")

        self.label_tab_2 = customtkinter.CTkLabel(tab_2, text="Backpropagation Network")
        self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)

        # Instantiate an object of the Backpropagation class
        self.backpropagation_model = Backpropagation(input_neurons=3, hidden_neurons=3, output_neurons=1)

        input_bp = StringVar()
        desired_output_bp = StringVar()
        learning_rate_bp = StringVar()

        def entry_input_bp():
            def parse_input_string(input_string):
                # Remove brackets and split the string into individual elements
                elements = input_string.replace("[", "").replace("]", "").split(",")

                # Convert elements to integers
                try:
                    elements = [int(element) for element in elements]
                except ValueError:
                    return None

                # Determine the sublist length based on the number of elements
                sublist_length = len(elements) // 4  # Assuming there are four sub lists in the input

                # Check if the number of elements is divisible by the calculated sublist_length
                if len(elements) % sublist_length != 0:
                    return None

                # Create sublist
                sub_lists = [elements[i:i + sublist_length] for i in range(0, len(elements), sublist_length)]

                return sub_lists

            def print_input_list(input_list):
                if input_list is not None:
                    print(input_list)
                else:
                    print("Invalid input format")

            def print_input_as_list(*args):
                input_value_str = input_bp.get()
                input_list_parsed = parse_input_string(input_value_str)
                print_input_list(input_list_parsed)

            self.entry_input_bp = customtkinter.CTkEntry(tab_2, placeholder_text="Input", textvariable=input_bp)
            self.entry_input_bp.grid(row=1, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            input_bp.trace("w", print_input_as_list)

        def entry_desired_output_bp():
            def parse_input_string(input_string):
                # Remove brackets and split the string into individual elements
                elements = input_string.replace("[", "").replace("]", "").split(",")

                # Convert elements to integers
                try:
                    elements = [int(element) for element in elements]
                except ValueError:
                    return None

                # Check if each element forms a single-element sublist
                sub_lists = [[element] for element in elements]

                return sub_lists

            def print_input_list(input_list):
                if input_list is not None:
                    print(input_list)
                else:
                    print("Invalid input format")

            def print_input_as_list(*args):
                input_value_str = desired_output_bp.get()
                input_list_parsed = parse_input_string(input_value_str)
                print_input_list(input_list_parsed)

            # Use desired_output as the text variable for the entry widget
            self.entry_desired_output_bp = customtkinter.CTkEntry(tab_2, placeholder_text="Desired output",
                                                                  textvariable=desired_output_bp)
            self.entry_desired_output_bp.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            desired_output_bp.trace("w", print_input_as_list)

        def entry_learning_rate_bp():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = learning_rate_bp.get()
                    # Convert the input value to a float
                    learning_rate_value = float(input_value_str)
                    # Use the learning_rate_value here
                    print("Learning rate:", learning_rate_value)
                except ValueError:
                    print("Invalid learning rate format")

            # Use learning_rate as the text variable for the entry widget
            self.entry_learning_rate_bp = customtkinter.CTkEntry(tab_2, placeholder_text="Learning rate",
                                                                 textvariable=learning_rate_bp)
            self.entry_learning_rate_bp.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            learning_rate_bp.trace("w", parse_entry)

        entry_input_bp()
        entry_desired_output_bp()
        entry_learning_rate_bp()

        def train_backprop():
            input_data_str = input_bp.get()  # This is a string; you'll need to convert it to a numpy array
            desired_output_str = desired_output_bp.get()  # Also a string to convert
            learning_rate_str = epoch_max_hop.get()  # And this is a string to convert to float

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

        self.button_train_bp = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                       text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                       command=train_backprop)

        self.button_train_bp.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        def test_backprop():
            input_data_str = input_bp.get()  # This is a string; you'll need to convert it to a numpy array
            desired_output_str = desired_output_bp.get()  # Also a string to convert

            # Convert the string representations to the appropriate types
            try:
                input_data = np.array(eval(input_data_str))  # Using eval can be risky; ensure the input is sanitized
                desired_output = np.array(eval(desired_output_str))

                # Now, you can use the data to train your neural network using the Backpropagation object
                self.backpropagation_model.test(input_data, desired_output)

                print("Test phase")
            except Exception as e:
                print(f"An error occurred: {e}")

        self.button_test_bp = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                      text="Test Network", text_color=("gray10", "#DCE4EE"),
                                                      command=test_backprop)
        self.button_test_bp.grid(row=4, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Plot loss
        def plot_loss_from_test():
            # Plot the loss directly using the loss values obtained during the test
            loss_values = self.backpropagation_model.loss

            # Clear the previous plot, if any
            self.canvas_tab_2.delete("all")

            # Clear the previous plot if it exists
            if hasattr(self, 'canvas'):
                self.canvas.get_tk_widget().destroy()
                plt.close(self.fig)

            # Create a new figure for the plot
            self.fig = plt.Figure(figsize=(4, 3))
            ax = self.fig.add_subplot(111)
            ax.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Mean Square Error')
            ax.set_title("Training loss")
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()

            # Embed the plot into the Tkinter canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_tab_2)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.canvas_tab_2 = customtkinter.CTkCanvas(tab_2, width=400, height=300)
        self.canvas_tab_2.grid(row=1, column=2, padx=(30, 0), pady=0)

        self.button_loss_bp = customtkinter.CTkButton(tab_2, fg_color="transparent", border_width=2,
                                                      text="Plot loss", text_color=("gray10", "#DCE4EE"),
                                                      command=plot_loss_from_test)
        self.button_loss_bp.grid(row=4, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # ########### 3rd Tab ###########
        tab_3 = self.tabview.tab("Kohonen SOM")

        self.label_tab_3 = customtkinter.CTkLabel(tab_3, text="Kohonen SOM Network")
        self.label_tab_3.grid(row=0, column=0, padx=20, pady=20)

        self.som_kohonen_model = SOM()

        num_of_neurons_som = StringVar()
        input_dim_som = StringVar()
        data_som = StringVar()
        lr_som = StringVar()
        epoch_max_som = StringVar()

        def entry1_tab3():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = input_dim_som.get()
                    # Convert the input value to a float
                    input_dim_value = int(input_value_str)
                    # Use the input_dim_value here
                    print("Input dimension:", input_dim_value)
                except ValueError:
                    print("Invalid input dimension format")

            self.entry1_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Input Dimension",
                                                       textvariable=input_dim_som)
            self.entry1_tab_3.grid(row=2, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            input_dim_som.trace("w", parse_entry)

        def entry2_tab3():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = num_of_neurons_som.get()
                    # Convert the input value to a float
                    num_of_neurons_value = int(input_value_str)
                    # Use the input_dim_value here
                    print("Number of neurons:", num_of_neurons_value)
                except ValueError:
                    print("Invalid number of neurons format")

            self.entry2_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Num of neurons",
                                                       textvariable=num_of_neurons_som)
            self.entry2_tab_3.grid(row=3, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            num_of_neurons_som.trace("w", parse_entry)

        def entry3_tab3():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = lr_som.get()
                    # Convert the input value to a float
                    lr_som_value = float(input_value_str)
                    # Use the input_dim_value here
                    print("Learning rate:", lr_som_value)
                except ValueError:
                    print("Invalid learning rate format")

            self.entry3_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Learning rate",
                                                       textvariable=lr_som)
            self.entry3_tab_3.grid(row=4, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            lr_som.trace("w", parse_entry)

        def entry4_tab3():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = epoch_max_som.get()
                    # Convert the input value to a float
                    epoch_max_value = int(input_value_str)
                    # Use the input_dim_value here
                    print("Epoch max:", epoch_max_value)
                except ValueError:
                    print("Invalid epoch max format")

            self.entry4_tab_3 = customtkinter.CTkEntry(tab_3, placeholder_text="Epoch max", textvariable=epoch_max_som)
            self.entry4_tab_3.grid(row=5, column=0, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            epoch_max_som.trace("w", parse_entry)

        def train_som():
            pass

        entry1_tab3()
        entry2_tab3()
        entry3_tab3()
        entry4_tab3()

        self.button1_tab_3 = customtkinter.CTkButton(tab_3, fg_color="transparent", border_width=2,
                                                     text="Train Network", text_color=("gray10", "#DCE4EE"))
        self.button1_tab_3.grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.button2_tab_3 = customtkinter.CTkButton(tab_3, fg_color="transparent", border_width=2,
                                                     text="Plot results", text_color=("gray10", "#DCE4EE"))
        self.button2_tab_3.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # ########### 4th Tab ###########
        tab_4 = self.tabview.tab("AutoEncoder")

        # Instantiate an object of the Backpropagation class
        self.autoencoder_model = AutoEncoder()

        input_ac = StringVar()
        learning_rate_ac = StringVar()
        momentum_ac = StringVar()
        epoch_max_ac = StringVar()

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

        # ########### 5th Tab ###########
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
