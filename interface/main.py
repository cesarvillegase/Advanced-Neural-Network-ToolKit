from tkinter import StringVar # filedialog as fd,
from PIL import Image, ImageTk
# from tkinter.messagebox import showinfo

import customtkinter
import matplotlib.pyplot as plt
import numpy as np
from customtkinter import CTkRadioButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neural_networks.hopfield import HopfieldNetwork
from neural_networks.backprop import Backpropagation
from neural_networks.som_kohonen import SOM
from neural_networks.autoencoder import AutoEncoder
from neural_networks.lvq import LvqNetwork


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
        logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="ANN ToolKit",
                                            font=customtkinter.CTkFont(size=20, weight="bold"))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Label for the levels of abstraction
        levels_of_abstraction_label = customtkinter.CTkLabel(self.sidebar_frame, text="Levels:",
                                                             anchor="w")
        levels_of_abstraction_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")

        # Variable to store the level of abstraction choice
        abstraction_level_var = customtkinter.StringVar(
            value="Low Level")  # Default value can be "Low Level" or "High Level"

        # Radio button for Low Level
        low_level_radio = CTkRadioButton(self.sidebar_frame, text="Low-Level",
                                         variable=abstraction_level_var, value="Low Level")
        low_level_radio.grid(row=2, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        # Radio button for High Level
        high_level_radio = CTkRadioButton(self.sidebar_frame, text="High-Level",
                                          variable=abstraction_level_var, value="High Level")
        high_level_radio.grid(row=3, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        # Create the label for the author and the description
        author_label = customtkinter.CTkLabel(self.sidebar_frame, text="Created by:",
                                              font=customtkinter.CTkFont(size=14, weight="normal"))
        author_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        author1_label = customtkinter.CTkLabel(self.sidebar_frame, text="Cesar A Villegas Espindola",
                                               font=customtkinter.CTkFont(size=14, weight="normal"))
        author1_label.grid(row=5, column=0, padx=20, pady=(10, 20))

        exit_button = customtkinter.CTkButton(self.sidebar_frame, fg_color="red", text="Close App",
                                              command=self.exit, anchor="w")
        exit_button.grid(row=6, column=0, padx=20, pady=(10, 20))

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
            if tab_name == "Kohonen SOM":
                self.setup_som_tab(tab)
            if tab_name == "AutoEncoder":
                self.setup_autoencoder(tab)
            if tab_name == "LVQ":
                self.setup_lvq(tab)

    # ########### 1st Tab ###########
    def setup_hopfield_tab(self, tab):
        label_hop = customtkinter.CTkLabel(tab, text="Hopfield Network", font=("bold", 24))
        label_hop.grid(row=0, column=0, padx=20, pady=20)

        epoch_max_hop = StringVar()

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

            label_epoch_max_hop = customtkinter.CTkLabel(tab, text="Epoch max:", font=("bold", 14))
            label_epoch_max_hop.grid(row=1, column=0, padx=20, pady=20)

            # Use learning_rate as the text variable for the entry widget
            entry_epoch_max_hop = customtkinter.CTkEntry(tab, textvariable=epoch_max_hop)
            entry_epoch_max_hop.grid(row=1, column=1, columnspan=1, padx=(20), pady=(20), sticky="nsew")

            # Attach the trace callback to the text variable
            epoch_max_hop.trace("w", parse_entry)

        entry_epoch_max_hop()

        # Images for the labels
        pil_image1 = Image.open(
            r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\labels\cat.jpg")
        resized_image1 = pil_image1.resize((240, 240))
        image1 = ImageTk.PhotoImage(resized_image1)

        pil_image2 = Image.open(
            r"C:\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\labels\fox.jpg")
        resized_image2 = pil_image2.resize((240, 240))
        image2 = ImageTk.PhotoImage(resized_image2)

        pil_image3 = Image.open(
            r"C:\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\labels\star.jpg")
        resized_image3 = pil_image3.resize((240, 240))
        image3 = ImageTk.PhotoImage(resized_image3)

        # select the image that you want to decode, Variable to store the image of choice
        image_choice_var = customtkinter.StringVar(value="First Image")  # Default is "First Image"

        # Radio buttons for selecting the Images
        first_image_radio = customtkinter.CTkRadioButton(tab, text="First Image",
                                                         variable=image_choice_var, value="First Image")
        first_image_radio.grid(row=4, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        first_image_label = customtkinter.CTkLabel(tab, image=image1, text="")
        first_image_label.grid(row=5, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_radio = customtkinter.CTkRadioButton(tab, text="Second Image",
                                                          variable=image_choice_var, value="Second Image")
        second_image_radio.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_label = customtkinter.CTkLabel(tab, image=image2, text="")
        second_image_label.grid(row=5, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_radio = customtkinter.CTkRadioButton(tab, text="Third Image",
                                                         variable=image_choice_var, value="Third Image")
        third_image_radio.grid(row=4, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_label = customtkinter.CTkLabel(tab, image=image3, text="")
        third_image_label.grid(row=5, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        # ######## OBTAIN THE IMAGE PATHS ########
        img_1_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\data\img_1.png"
        img_2_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\data\img_2.png"
        img_3_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\data\img_3.png"

        img_1_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\data\img_1_noisy.png"
        img_2_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\data\img_2_noisy.png"
        img_3_wn_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\hopfield\data\img_3_noisy.png"

        # ######## LOAD THE IMAGES ########
        img_1 = Image.open(img_1_path)
        img_2 = Image.open(img_2_path)
        img_3 = Image.open(img_3_path)

        img_1_wn = Image.open(img_1_wn_path)
        img_2_wn = Image.open(img_2_wn_path)
        img_3_wn = Image.open(img_3_wn_path)

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
        imgs = [original_img[0], noisy_img[0],
                reconstructed_img[0]]  # Access the first element since each is wrapped in a list
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
        backpropagation_model = Backpropagation(input_neurons=3, hidden_neurons=3, output_neurons=1)

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
                backpropagation_model.train(input_data, desired_output, learning_rate)

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
                backpropagation_model.test(input_data, desired_output)

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
            loss_values = backpropagation_model.loss

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

    # ########### 3rd Tab ###########
    def setup_som_tab(self, tab):
        label_tab_3 = customtkinter.CTkLabel(tab, text="Kohonen SOM Network", font=("bold", 24))
        label_tab_3.grid(row=0, column=0, padx=20, pady=20)

        som_model = SOM()

        num_point_per_class_som = StringVar()
        num_classes_som = StringVar()
        num_of_neurons_som = StringVar()
        input_dim_som = StringVar()
        lr_som = StringVar()
        epoch_max_som = StringVar()

        def entry_num_points_per_class_som():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = num_point_per_class_som.get()
                    # Convert the input value to a float
                    num_points_per_class_value = int(input_value_str)
                    # Use the input_dim_value here
                    print("Number of points per class:", num_points_per_class_value)
                except ValueError:
                    print("Invalid number of points per class format")

            label_num_points_pclass = customtkinter.CTkLabel(tab, text="Num. of points per class:")
            label_num_points_pclass.grid(row=1, column=0, padx=20, pady=20)

            entry_num_points_per_class_som = customtkinter.CTkEntry(tab, placeholder_text="Number of points per class",
                                                                    textvariable=num_point_per_class_som)
            entry_num_points_per_class_som.grid(row=1, column=1, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            num_point_per_class_som.trace("w", parse_entry)

        def entry_classes_som():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = num_classes_som.get()
                    # Convert the input value to a float
                    num_classes_som_value = int(input_value_str)
                    # Use the input_dim_value here
                    print("Number of classes:", num_classes_som_value)
                except ValueError:
                    print("Invalid Number of classes format")

            label_num_classes_som = customtkinter.CTkLabel(tab, text="Num. of classes:")
            label_num_classes_som.grid(row=1, column=2, padx=20, pady=20)

            entry_num_classes_som = customtkinter.CTkEntry(tab, textvariable=num_classes_som)
            entry_num_classes_som.grid(row=1, column=3, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            num_classes_som.trace("w", parse_entry)

        def entry_number_of_neurons_som():
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

            label_number_of_neurons_som = customtkinter.CTkLabel(tab, text="Num of neurons:")
            label_number_of_neurons_som.grid(row=2, column=0, padx=20, pady=20)

            entry_number_of_neurons_som = customtkinter.CTkEntry(tab, textvariable=num_of_neurons_som)
            entry_number_of_neurons_som.grid(row=2, column=1, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            num_of_neurons_som.trace("w", parse_entry)

        def entry_input_dimension_som():
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

            label_input_dimension = customtkinter.CTkLabel(tab, text="Input dimension:")
            label_input_dimension.grid(row=2, column=2, padx=20, pady=20)

            entry_input_dimension_som = customtkinter.CTkEntry(tab, textvariable=input_dim_som)
            entry_input_dimension_som.grid(row=2, column=3, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            input_dim_som.trace("w", parse_entry)

        def entry_learning_rate_som():
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

            label_learning_rate_som = customtkinter.CTkLabel(tab, text="Learning rate")
            label_learning_rate_som.grid(row=3, column=0, padx=20, pady=20)

            entry_learning_rate_som = customtkinter.CTkEntry(tab, textvariable=lr_som)
            entry_learning_rate_som.grid(row=3, column=1, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            lr_som.trace("w", parse_entry)

        def entry_epoch_max_som():
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

            label_epoch_max_som = customtkinter.CTkLabel(tab, text="Epoch max:")
            label_epoch_max_som.grid(row=3, column=2, padx=20, pady=20)

            entry_epoch_max_som = customtkinter.CTkEntry(tab, textvariable=epoch_max_som)
            entry_epoch_max_som.grid(row=3, column=3, padx=(20, 0), pady=(20, 20), sticky="nsew")

            # Attach the trace callback to the text variable
            epoch_max_som.trace("w", parse_entry)

        entry_num_points_per_class_som()
        entry_classes_som()
        entry_number_of_neurons_som()
        entry_input_dimension_som()
        entry_learning_rate_som()
        entry_epoch_max_som()

        def setup_data_som(num_points_p_class, num_classes):
            np.random.seed(42)
            data = []
            labels = []

            for i in range(num_classes):
                points = np.random.rand(num_points_p_class, 2) * 2

                if i == 1:
                    points += np.array([3, 3])
                elif i == 2:
                    points += np.array([0, 4])
                elif i == 3:
                    points += np.array([3, 0])

                data.append(points)
                labels.append(np.full(num_points_p_class, i))

            data = np.vstack(data)
            y = np.concatenate(labels)

            print("The data is generated")

            return data, y

        def generate_data():
            num_points_p_class_str = num_point_per_class_som.get()  # From string to a numpy array
            num_classes_str = num_classes_som.get()

            # Convert the string representations to the appropriate types
            try:
                num_points_p_class = int(num_points_p_class_str)
                num_classes = int(num_classes_str)

                # Get the data generated by the generate_data() function
                self.data, self.labels = setup_data_som(num_points_p_class, num_classes)

            except Exception as e:
                print(f"An error occurred: {e}")

        pil_image_random_data = Image.open(
            r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\som\plot\random_data.jpeg")
        resized_image_rand_data = pil_image_random_data.resize((180, 180))
        image_rand_data = ImageTk.PhotoImage(resized_image_rand_data)

        label_rand_data = customtkinter.CTkLabel(tab, text="Random Data", font=("bold", 18))
        label_rand_data.grid(row=2, column=4, padx=(20, 0), pady=(5, 5), sticky="w")

        image_label_rand_data = customtkinter.CTkLabel(tab, image=image_rand_data, text="")
        image_label_rand_data.grid(row=3, column=4, padx=(20, 0), pady=(5, 5), sticky="w")

        def train_som():
            num_neurons_str = num_of_neurons_som.get()
            input_dim_str = input_dim_som.get()
            learning_rate_str = lr_som.get()
            epoch_max_str = epoch_max_som.get()

            # Convert the string representations to the appropriate types
            try:
                num_of_neurons = int(num_neurons_str)
                input_dim = int(input_dim_str)
                learning_rate = float(learning_rate_str)
                epoch_max = int(epoch_max_str)

                print("Training phase")

                # Now, you can use the data to train your neural network usingas the SOM model
                self.trained_weights = som_model.train(num_of_neurons, input_dim, self.data, learning_rate, epoch_max)
                print("Training completed.")

            except Exception as e:
                print(f"An error occurred: {e}")

        def plot_som():
            fig, ax = plt.subplots(figsize=(4, 4))

            # Plot training data
            ax.scatter(self.data[:, 0], self.data[:, 1], c='r', marker='x', label='Training Data')

            # Plot trained weights
            ax.scatter(self.trained_weights[:, 0], self.trained_weights[:, 1], c='b', marker='o', label='Neurons')

            ax.set_title("Trained weights")
            ax.legend()

            plt.show()

        button_generate_data_som = customtkinter.CTkButton(tab, fg_color="#219ebc", border_width=2,
                                                           text="Generate data", text_color="white",
                                                           command=generate_data)
        # command=generate_data_som()) num_points_p_class=, num_classes=
        button_generate_data_som.grid(row=7, column=0, padx=(20, 20), pady=(20, 20))

        button_train_network = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                       text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                       command=train_som)
        button_train_network.grid(row=7, column=1, padx=(20, 20), pady=(20, 20))

        button_plot_results = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                      text="Plot results", text_color=("gray10", "#DCE4EE"),
                                                      command=plot_som)
        button_plot_results.grid(row=7, column=2, padx=(20, 20), pady=(20, 20))

    # ########### 4th Tab ###########
    def setup_autoencoder(self, tab):

        learning_rate_ac = StringVar()
        momentum_ac = StringVar()
        epoch_max_ac = StringVar()

        label_tab = customtkinter.CTkLabel(tab, text="AutoEncoder Network", font=("bold",  24))
        label_tab.grid(row=0, column=0, padx=20, pady=20)

        def entry_learning_rate_ac():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = float(learning_rate_ac.get())
                    # Convert the input value to an integer
                    learning_rate_ac_value = float(input_value_str)
                    # Use the learning_rate_value here
                    print("Learning rate:", learning_rate_ac_value)
                except ValueError:
                    print("Invalid Learning rate format")

            label_learning_rate = customtkinter.CTkLabel(tab, text="Learning rate")
            label_learning_rate.grid(row=1, column=0, padx=(20, 0), pady=(20, 20))

            entry_learning_rate = customtkinter.CTkEntry(tab, textvariable=learning_rate_ac)
            entry_learning_rate.grid(row=1, column=1, padx=(20, 0), pady=(20, 20))  # , sticky="nsew"
            # Attach the trace callback to the text variable
            learning_rate_ac.trace("w", parse_entry)

        def entry_momentum_rate_ac():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = float(momentum_ac.get())
                    # Convert the input value to an integer
                    momentum_ac_value = float(input_value_str)
                    # Use the learning_rate_value here
                    print("Momentum rate:", momentum_ac_value)
                except ValueError:
                    print("Invalid Momentum rate format")

            label_momentum_rate = customtkinter.CTkLabel(tab, text="Momentum rate")
            label_momentum_rate.grid(row=1, column=2, padx=(20, 0), pady=(20, 20))  # , sticky="nsew"

            entry_momentum_rate = customtkinter.CTkEntry(tab, textvariable=momentum_ac)
            entry_momentum_rate.grid(row=1, column=3, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            momentum_ac.trace("w", parse_entry)

        def entry_epoch_max_ac():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = int(epoch_max_ac.get())  # Corrected to get epoch max, not momentum
                    # Convert the input value to an integer
                    epoch_max_ac_value = int(input_value_str)
                    print("Epoch Max:", epoch_max_ac_value)
                except ValueError:
                    print("Invalid Epoch Max format")

            label_epoch_max = customtkinter.CTkLabel(tab, text="Epoch max")
            label_epoch_max.grid(row=2, column=0, padx=(20, 0), pady=(20, 20))

            entry_epoch_max = customtkinter.CTkEntry(tab, textvariable=epoch_max_ac)
            entry_epoch_max.grid(row=2, column=1, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            epoch_max_ac.trace("w", parse_entry)

        entry_learning_rate_ac()
        entry_momentum_rate_ac()
        entry_epoch_max_ac()

        # Images for the labels
        pil_image1 = Image.open(
            r"C:\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\autoencoder\labels\rat.jpeg")
        resized_image1 = pil_image1.resize((240, 240))
        image1 = ImageTk.PhotoImage(resized_image1)

        pil_image2 = Image.open(
            r"C:\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\autoencoder\labels\chicken.jpeg")
        resized_image2 = pil_image2.resize((240, 240))
        image2 = ImageTk.PhotoImage(resized_image2)

        pil_image3 = Image.open(
            r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\autoencoder\labels\cat.jpeg")
        resized_image3 = pil_image3.resize((240, 240))
        image3 = ImageTk.PhotoImage(resized_image3)

        # select the image that you want to decode, Variable to store the image of choice
        image_choice_var = customtkinter.StringVar(value="First Image")  # Default is "First Image"

        # Radio buttons for selecting the Images
        first_image_radio = customtkinter.CTkRadioButton(tab, text="First Image",
                                                         variable=image_choice_var, value="First Image")
        first_image_radio.grid(row=3, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        first_image_label = customtkinter.CTkLabel(tab, image=image1, text="")
        first_image_label.grid(row=4, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_radio = customtkinter.CTkRadioButton(tab, text="Second Image",
                                                          variable=image_choice_var, value="Second Image")
        second_image_radio.grid(row=3, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_label = customtkinter.CTkLabel(tab, image=image2, text="")
        second_image_label.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_radio = customtkinter.CTkRadioButton(tab, text="Third Image",
                                                         variable=image_choice_var, value="Third Image")
        third_image_radio.grid(row=3, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_label = customtkinter.CTkLabel(tab, image=image3, text="")
        third_image_label.grid(row=4, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        # ######## OBTAIN THE IMAGE PATHS ########
        img_1_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\autoencoder\data\img_1.png"
        img_2_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\autoencoder\data\img_2.png"
        img_3_path = r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\autoencoder\data\img_3.png"

        # ######## LOAD THE IMAGES ########
        img_1 = Image.open(img_1_path).convert("RGB")
        img_2 = Image.open(img_2_path).convert("RGB")
        img_3 = Image.open(img_3_path).convert("RGB")

        # ######## NORMALIZE THE IMAGES ########
        img_1_array = np.array(img_1) # / 255.0 * 2 - 1
        img_2_array = np.array(img_2) # / 255.0 * 2 - 1
        img_3_array = np.array(img_3) # / 255.0 * 2 - 1

        # ######## Function to obtain the choosen image ########
        def get_selected_image_data():
            selected_value = image_choice_var.get()
            if selected_value == "First Image":
                return img_1_array
            elif selected_value == "Second Image":
                return img_2_array
            elif selected_value == "Third Image":
                return img_3_array
            else:
                return None

        autoencoder_model = AutoEncoder()

        # loss_ac, latent_space, decoded_inputs_ac = model.train(X_train_ac, alpha_ac, momentum_ac, epoch_max_ac)

        def train_ac():
            data = get_selected_image_data()
            if data is not None:
                learning_rate_ac_value = float(learning_rate_ac.get())
                momentum_ac_value = float(momentum_ac.get())
                epoch_max_ac_value = int(epoch_max_ac.get())

                self.loss, latent_space, self.decoded_inputs = autoencoder_model.train(data, learning_rate_ac_value, momentum_ac_value, epoch_max_ac_value)
            else:
                print("The model need's the hyperparameters")

        # Modify the function call to plot_images
        def reconstruct_selected_image():
            data = get_selected_image_data()
            if data is not None:
                reconstructed_image = self.decoded_inputs.reshape(16, 16, 3)  # This should be a NumPy array. If it's not, you'll have to adjust this part.
                if isinstance(reconstructed_image, Image.Image):
                    # If decoded_inputs is still a PIL Image, convert it to a NumPy array
                    reconstructed_image = Image.fromarray(np.uint8(self.decoded_inputs))

                original_img = [(data)]

                self.plot_images(original_img, reconstructed_image)

        def plot_loss_from_training():
            # Plot the loss directly using the loss values obtained during the test
            loss_values = self.loss

            plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference
            plt.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Mean Square Error')
            plt.title("Training loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.show()


        button_train_ac = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                command=train_ac)
        button_train_ac.grid(row=6, column=0, padx=(20, 20), pady=(20, 20))

        button_reconstruction_ac = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                            text="Reconstruct image",
                                                            text_color=("gray10", "#DCE4EE"),
                                                            anchor="w", command=reconstruct_selected_image)
        button_reconstruction_ac.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        button_plot_loss_ac = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                text="Plot loss", text_color=("gray10", "#DCE4EE"),
                                                      command=plot_loss_from_training)
        button_plot_loss_ac.grid(row=6, column=2, padx=(20, 20), pady=(20, 20))

    def plot_images(self, original_image, reconstructed_img):
        """Plot the original, noisy, and reconstructed images."""
        plt.figure(figsize=(8, 4))

        # Reshape and convert the original image to a NumPy array
        original_image_array = np.array(original_image[0])
        original_image_array = original_image_array.astype(np.uint8) #+ 1) / 2 * 255

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image_array)
        plt.title('Original Image')
        plt.axis('off')  # Turn off axes

        # Plot the reconstructed image
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_img)
        plt.title('Reconstructed Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # ########### 5th Tab ###########
    def setup_lvq(self, tab):
        label_lvq = customtkinter.CTkLabel(tab, text="LVQ Network", font=("bold", 24))
        label_lvq.grid(row=0, column=0, padx=20, pady=20)

        input_lvq = StringVar()
        labels_lvq = StringVar()
        learning_rate_lvq = StringVar()
        epoch_max_lvq = StringVar()

        def entry_input_lvq():
            def parse_input_string(input_string):
                # Remove unnecessary characters and split the string into individual elements
                elements = input_string.replace("[", "").replace("]", "").split(",")

                # Convert elements to floats
                try:
                    elements = [float(element.strip()) for element in elements]
                except ValueError:
                    return None

                # Check if the number of elements is divisible by 2 (assuming each data point has two coordinates)
                if len(elements) % 2 != 0:
                    return None

                # Create sublists of coordinates
                sub_lists = [elements[i:i + 2] for i in range(0, len(elements), 2)]

                return sub_lists

            def print_input_list(input_list):
                if input_list is not None:
                    print(input_list)
                    # You can perform further processing or pass this data to your LVQ algorithm
                else:
                    print("Invalid input format")

            def print_input_as_list(*args):
                input_value_str = input_lvq.get()
                input_list_parsed = parse_input_string(input_value_str)
                print_input_list(input_list_parsed)

            label_input = customtkinter.CTkLabel(tab, text="Input")
            label_input.grid(row=1, column=0, padx=(20, 0), pady=(20, 20))

            entry_input = customtkinter.CTkEntry(tab, textvariable=input_lvq)
            entry_input.grid(row=1, column=1, padx=(20, 0), pady=(20, 20))

            input_lvq.trace("w", print_input_as_list)

        def entry_labels_lvq():
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
                input_value_str = labels_lvq.get()
                input_list_parsed = parse_input_string(input_value_str)
                print_input_list(input_list_parsed)

            label_labels_lvq = customtkinter.CTkLabel(tab, text="Desired output")
            label_labels_lvq.grid(row=1, column=2, padx=(20, 0), pady=(20,0))

            # Use desired_output as the text variable for the entry widget
            entry_desired_output_bp = customtkinter.CTkEntry(tab, textvariable=labels_lvq)
            entry_desired_output_bp.grid(row=1, column=3, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            labels_lvq.trace("w", print_input_as_list)

        def entry_learning_rate_lvq():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = learning_rate_lvq.get()
                    # Convert the input value to a float
                    learning_rate_value = float(input_value_str)
                    # Use the learning_rate_value here
                    print("Learning rate:", learning_rate_value)
                except ValueError:
                    print("Invalid learning rate format")

            label_learning_rate_lvq = customtkinter.CTkLabel(tab, text="Learning rate", font=("bold", 14))
            label_learning_rate_lvq.grid(row=2, column=0, padx=20, pady=20)

            # Use learning_rate as the text variable for the entry widget
            entry_learning_rate_lvq = customtkinter.CTkEntry(tab, textvariable=learning_rate_lvq)
            entry_learning_rate_lvq.grid(row=2, column=1, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            learning_rate_lvq.trace("w", parse_entry)

        def entry_epoch_max_lvq():
            def parse_entry(*args):
                try:
                    # Get the input value from the entry widget
                    input_value_str = int(epoch_max_lvq.get())
                    # Convert the input value to an integer
                    epoch_max_hop_value = int(input_value_str)
                    # Use the learning_rate_value here
                    print("Epoch max:", epoch_max_hop_value)
                except ValueError:
                    print("Invalid Epoch max format")

            label_epoch_max_lvq = customtkinter.CTkLabel(tab, text="Epoch max:", font=("bold", 14))
            label_epoch_max_lvq.grid(row=2, column=2, padx=20, pady=20)

            # Use learning_rate as the text variable for the entry widget
            entry_epoch_max_lvq = customtkinter.CTkEntry(tab, textvariable=epoch_max_lvq)
            entry_epoch_max_lvq.grid(row=2, column=3, padx=(20), pady=(20))

            # Attach the trace callback to the text variable
            epoch_max_lvq.trace("w", parse_entry)

        entry_input_lvq()
        entry_labels_lvq()
        entry_learning_rate_lvq()
        entry_epoch_max_lvq()

        lvq_model = LvqNetwork()

        def train_lvq():
            input_data_str = input_lvq.get()
            desired_output_str = labels_lvq.get()
            learning_rate_str = learning_rate_lvq.get()
            epoch_max_lvq_str = epoch_max_lvq.get()

            try:
                input_data = np.array(eval(input_data_str))
                desired_output = np.array(eval(desired_output_str))
                learning_rate = float(learning_rate_str)
                epoch_max = int(epoch_max_lvq_str)

                trained_vectors = lvq_model.train(input_data, desired_output, learning_rate, epoch_max)

                print("Training completed")

                # Plot the trained vectors and data
                plot(input_data, trained_vectors, desired_output, "Trained Vectors and Data", test=True)
            except Exception as e:
                print(f"An error occurred: {e}")

        button_train_ac = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                  text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                  command=train_lvq)
        button_train_ac.grid(row=3, column=0, padx=(20, 20), pady=(20, 20))

        '''
        button_test_lvq = customtkinter.CTkButton(tab, fg_color="transparent", border_width=2,
                                                           text="Reconstruct image",
                                                           text_color=("gray10", "#DCE4EE"),
                                                           command=test_lvq)
        button_test_lvq.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        '''

    def plot(data, vectors, labels, title):
        with plt.style.context('seaborn-darkgrid'):
            plt.figure(figsize=(8, 6))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', label='Data', alpha=0.6)
            plt.scatter(vectors[:, 0], vectors[:, 1], c='red', marker='x', s=100, label='Vectors')

            for i, label in enumerate(labels):
                plt.text(data[i, 0], data[i, 1], str(label), color='black', fontsize=10, ha='center', va='center')

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()

    def exit(self):
        self.destroy()


'''     


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

if __name__ == "__main__":
    app = App()
    app.mainloop()
