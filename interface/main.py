from tkinter import filedialog as fd, StringVar
from PIL import Image, ImageTk
from tkinter.messagebox import showinfo

import customtkinter
import matplotlib.pyplot as plt
import numpy as np
from customtkinter import CTkRadioButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neural_networks.hopfield import HopfieldNetwork
from neural_networks.backprop import Backpropagation
from neural_networks.som_kohonen import SOM
from neural_networks.autoencoder import AutoEncoder

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.configure_window()
        self.create_sidebar_frame()
        self.create_tabview()

    def configure_window(self):
        # Configure window
        self.title("Advanced Neural Network ToolKit")
        self.geometry(f"{1280}x{640}")
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

    def create_sidebar_frame(self):
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.setup_sidebar_widgets()

    def setup_sidebar_widgets(self):
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

    def create_tabview(self):
        # Create tabview
        tabview = customtkinter.CTkTabview(self, width=250)
        tabview.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.setup_tabs(tabview)

    def setup_tabs(self, tabview):
        tabs = ["Hopfield", "Backpropagation", "Kohonen SOM", "AutoEncoder", "LVQ"]
        for tab_name in tabs:
            tab = tabview.add(tab_name)
            if tab_name == "Hopfield":
                self.setup_hopfield_tab(tab)
            if tab_name == "Backpropagation":
                self.setup_backprop_tab(tab)
            '''
            if tab_name == "Kohonen SOM":
                self.setup_som_tab(tab)
            if tab_name == "AutoEncoder":
                self.setup_som_tab(tab)
            if tab_name == "LVQ":
                self.setup_lvq_tab(tab)

            '''

    # ########### 1st Tab ###########
    def setup_hopfield_tab(self, tab):
        label_hop = customtkinter.CTkLabel(tab, text="Hopfield Network", font=("bold", 24))
        label_hop.grid(row=0, column=0, padx=20, pady=20)

        epoch_max_hop = StringVar()

        label_epoch_max_hop = customtkinter.CTkLabel(tab, text="Epoch max:", font=("bold", 14))
        label_epoch_max_hop.grid(row=1, column=0, padx=20, pady=20)

        def entry_epoch_max_hop():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = int(epoch_max_hop.get())
                    # Convert the input value to an integer
                    epoch_max_hop_value = int(input_value_str)
                    # Use the learning_rate_value here
                    print("Epoch max:", epoch_max_hop_value)
                except ValueError:
                    print("Invalid Epoch max format")

            # Use learning_rate as the text variable for the entry widget
            self.entry_epoch_max_hop = customtkinter.CTkEntry(tab, placeholder_text="Epoch max",
                                                              textvariable=epoch_max_hop)
            self.entry_epoch_max_hop.grid(row=1, column=1, columnspan=1, padx=(20), pady=(20), sticky="nsew")

            # Attach the trace callback to the text variable
            epoch_max_hop.trace("w", parse_entry)

        entry_epoch_max_hop()

        # Images for the labels
        pil_image1 = Image.open(
            r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hop_labels\cat.jpg")
        resized_image1 = pil_image1.resize((240, 240))
        self.image1 = ImageTk.PhotoImage(resized_image1)

        pil_image2 = Image.open(
            r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hop_labels\fox.jpg")
        resized_image2 = pil_image2.resize((240, 240))
        self.image2 = ImageTk.PhotoImage(resized_image2)

        pil_image3 = Image.open(
            r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hop_labels\star.jpg")
        resized_image3 = pil_image3.resize((240, 240))
        self.image3 = ImageTk.PhotoImage(resized_image3)

        # select the image that you want to decode, Variable to store the image of choice
        image_choice_var = customtkinter.StringVar(value="First Image")  # Default is "First Image"

        # Radio buttons for selecting the Images
        first_image_radio = customtkinter.CTkRadioButton(tab, text="First Image",
                                                         variable=image_choice_var, value="First Image")
        first_image_radio.grid(row=4, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        first_image_label = customtkinter.CTkLabel(tab, image=self.image1, text="")
        first_image_label.grid(row=5, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_radio = customtkinter.CTkRadioButton(tab, text="Second Image",
                                                          variable=image_choice_var, value="Second Image")
        second_image_radio.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_label = customtkinter.CTkLabel(tab, image=self.image2, text="")
        second_image_label.grid(row=5, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_radio = customtkinter.CTkRadioButton(tab, text="Third Image",
                                                         variable=image_choice_var, value="Third Image")
        third_image_radio.grid(row=4, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_label = customtkinter.CTkLabel(tab, image=self.image3, text="")
        third_image_label.grid(row=5, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        # ######## OBTAIN THE IMAGE PATHS ########
        img_1_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_1.png"
        img_2_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_2.png"
        img_3_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_3.png"

        img_1_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_1_noisy.png"
        img_2_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_2_noisy.png"
        img_3_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\img_3_noisy.png"

        # ######## LOAD THE IMAGES ########
        img_1 = Image.open(img_1_path)
        img_2 = Image.open(img_2_path)
        img_3 = Image.open(img_3_path)

        img_1_wn = Image.open(img_1_wn_path)
        img_2_wn = Image.open(img_2_wn_path)
        img_3_wn = Image.open(img_3_wn_path)

        # Make copies of the original images
        img_1_original = img_1.copy()
        img_2_original = img_2.copy()
        img_3_original = img_3.copy()

        img_1_wn_original = img_1_wn.copy()
        img_2_wn_original = img_2_wn.copy()
        img_3_wn_original = img_3_wn.copy()

        # ######## NORMALIZE THE IMAGES ########
        img_1_array = np.array(img_1) / 255.0 * 2 - 1
        img_2_array = np.array(img_2) / 255.0 * 2 - 1
        img_3_array = np.array(img_3) / 255.0 * 2 - 1

        img_1_wn_array = np.array(img_1_wn) / 255.0 * 2 - 1
        img_2_wn_array = np.array(img_2_wn) / 255.0 * 2 - 1
        img_3_wn_array = np.array(img_3_wn) / 255.0 * 2 - 1

        # ######## Function to obtain the choosen image ########
        def get_selected_image_data():
            selected_value = image_choice_var.get()
            if selected_value == "First Image":
                return img_1_array, img_1_wn_array
            elif selected_value == "Second Image":
                return img_2_array, img_2_wn_array
            elif selected_value == "Third Image":
                return img_3_array, img_3_wn_array
            else:
                return None, None

        # train the network
        def train_hopfield():
            data, noisy_data = get_selected_image_data()
            if data is not None and noisy_data is not None:
                if epoch_max_hop.get().isdigit():
                    epoch_max_hop_value = int(epoch_max_hop.get())
                    self.hopfield_model = HopfieldNetwork(epoch_max=epoch_max_hop_value)
                    self.hopfield_model.train([data])
                else:
                    print("Epoch max needs to be an integer value.")

        # Modify the function call to plot_images
        def reconstruct_selected_image():
            data, noisy_data = get_selected_image_data()
            if data is not None:
                reconstructed_image = self.hopfield_model.reconstruct(noisy_data)
                # Plot the original, noisy, and reconstructed images
                original_img = [((data + 1) / 2 * 255).astype(np.uint8)]
                noisy_img = [((noisy_data + 1) / 2 * 255).astype(np.uint8)]
                rec_img = [((reconstructed_image + 1) / 2 * 255).astype(np.uint8)]
                self.plot_images(original_img, noisy_img, rec_img)

        # Create buttons for training and reconstruction
        button_train_hop = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                   text="Train algorithm",
                                                   text_color=("gray10", "#DCE4EE"),
                                                   anchor="w", command=train_hopfield)
        button_train_hop.grid(row=6, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

        button_reconstruction_hop = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                            text="Reconstruct image",
                                                            text_color=("gray10", "#DCE4EE"),
                                                            anchor="w", command=reconstruct_selected_image)
        button_reconstruction_hop.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Create a canvas for the reconstructed images
        canvas_plot = customtkinter.CTkCanvas(tab, width=0, height=0)
        canvas_plot.grid(row=3, column=4, padx=0, pady=0)

    def plot_images(self, original_img, noisy_img, reconstructed_img):
        """Plot the original, noisy, and reconstructed images."""
        plt.figure(figsize=(12, 4))
        imgs = [original_img[0], noisy_img[0], reconstructed_img[0]]  # Access the first element since each is wrapped in a list
        titles = ['Original Image', 'Noisy Image', 'Reconstructed Image']
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(imgs[i].astype(np.uint8))
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()


    # ########### 2nd Tab ###########
    def setup_backprop_tab(self, tab):
        label_tab_2 = customtkinter.CTkLabel(tab, text="Backpropagation Network")
        label_tab_2.grid(row=0, column=0, padx=20, pady=20)

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

            self.entry_input_bp = customtkinter.CTkEntry(tab, placeholder_text="Input", textvariable=input_bp)
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
            self.entry_desired_output_bp = customtkinter.CTkEntry(tab, placeholder_text="Desired output",
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
            self.entry_learning_rate_bp = customtkinter.CTkEntry(tab, placeholder_text="Learning rate",
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
            learning_rate_str = learning_rate_bp.get()  # And this is a string to convert to float

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

        button_train_bp = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                  text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                  command=train_backprop)

        button_train_bp.grid(row=4, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")

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

        button_test_bp = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                 text="Test Network", text_color=("gray10", "#DCE4EE"),
                                                 command=test_backprop)
        button_test_bp.grid(row=4, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Plot loss
        def plot_loss_from_test():
            # Plot the loss directly using the loss values obtained during the test
            loss_values = self.backpropagation_model.loss

            # Clear the previous plot, if any
            canvas_tab_2.delete("all")

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
            self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_tab_2)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)

        canvas_tab_2 = customtkinter.CTkCanvas(tab, width=400, height=300)
        canvas_tab_2.grid(row=1, column=2, padx=(30, 0), pady=0)

        button_loss_bp = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                 text="Plot loss", text_color=("gray10", "#DCE4EE"),
                                                 command=plot_loss_from_test)
        button_loss_bp.grid(row=4, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        '''
        

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
        '''

    def exit(self):
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
