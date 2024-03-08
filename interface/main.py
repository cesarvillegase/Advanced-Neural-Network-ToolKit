from tkinter import StringVar  # filedialog as fd,
from PIL import Image, ImageTk
# from tkinter.messagebox import showinfo

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from customtkinter import CTkRadioButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from neural_networks.hopfield import HopfieldNetwork
from neural_networks.backprop import Backpropagation
from neural_networks.som_kohonen import SOM
from neural_networks.autoencoder import AutoEncoder
from neural_networks.lvq import LvqNetwork

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
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
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.setup_sidebar_widgets()

    def setup_sidebar_widgets(self):
        # Create a logo label inside the sidebar
        logo_label = ctk.CTkLabel(self.sidebar_frame, text="ANN ToolKit",
                                            font=ctk.CTkFont(size=20, weight="bold"))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Label for the levels of abstraction
        levels_of_abstraction_label = ctk.CTkLabel(self.sidebar_frame, text="Levels:",
                                                             anchor="w")
        levels_of_abstraction_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")

        # Variable to store the level of abstraction choice
        abstraction_level_var = ctk.StringVar(
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
        author_label = ctk.CTkLabel(self.sidebar_frame, text="Created by:",
                                              font=ctk.CTkFont(size=14, weight="normal"))
        author_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        author1_label = ctk.CTkLabel(self.sidebar_frame, text="Cesar A Villegas Espindola",
                                               font=ctk.CTkFont(size=14, weight="normal"))
        author1_label.grid(row=5, column=0, padx=20, pady=(10, 20))

        exit_button = ctk.CTkButton(self.sidebar_frame, fg_color="red", text="Close App",
                                              command=self.exit, anchor="w")
        exit_button.grid(row=6, column=0, padx=20, pady=(10, 20))

    def create_tabview(self):
        # Create tabview
        tabview = ctk.CTkTabview(self, width=250)
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
        label_hop = ctk.CTkLabel(tab, text="Hopfield Network", font=("bold", 24))
        label_hop.grid(row=0, column=0, padx=20, pady=20, sticky="w")

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

            label_epoch_max_hop = ctk.CTkLabel(tab, text="Epoch max:", font=("bold", 14))
            label_epoch_max_hop.grid(row=1, column=0, padx=20, pady=20)

            # Use learning_rate as the text variable for the entry widget
            entry_epoch_max_hop = ctk.CTkEntry(tab, textvariable=epoch_max_hop)
            entry_epoch_max_hop.grid(row=1, column=1, padx=(20), pady=(20))

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
        image_choice_var = ctk.StringVar(value="First Image")  # Default is "First Image"

        # Radio buttons for selecting the Images
        first_image_radio = ctk.CTkRadioButton(tab, text="First Image",
                                                         variable=image_choice_var, value="First Image")
        first_image_radio.grid(row=4, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        first_image_label = ctk.CTkLabel(tab, image=image1, text="")
        first_image_label.grid(row=5, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_radio = ctk.CTkRadioButton(tab, text="Second Image",
                                                          variable=image_choice_var, value="Second Image")
        second_image_radio.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_label = ctk.CTkLabel(tab, image=image2, text="")
        second_image_label.grid(row=5, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_radio = ctk.CTkRadioButton(tab, text="Third Image",
                                                         variable=image_choice_var, value="Third Image")
        third_image_radio.grid(row=4, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_label = ctk.CTkLabel(tab, image=image3, text="")
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

        # ######## Function to obtain the chosen image ########
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
                    print("Training phase completed")
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
                self.plot_images_hop(original_img, noisy_img, rec_img)
                print("Reconstruction phase completed")

        # Create buttons for training and reconstruction
        button_train_hop = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                   text="Train algorithm",
                                                   text_color=("gray10", "#DCE4EE"),
                                                   anchor="w", command=train_hopfield)
        button_train_hop.grid(row=6, column=0, padx=(20, 20), pady=(20, 20))

        button_reconstruction_hop = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                            text="Reconstruct image",
                                                            text_color=("gray10", "#DCE4EE"),
                                                            anchor="w", command=reconstruct_selected_image)
        button_reconstruction_hop.grid(row=6, column=1, padx=(20, 20), pady=(20, 20))

    def plot_images_hop(self, original_img, noisy_img, reconstructed_img):
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
        label_tab_backprop = ctk.CTkLabel(tab, text="Backpropagation Network", font=("bold", 24))
        label_tab_backprop.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Instantiate an object of the Backpropagation class
        backpropagation_model = Backpropagation(input_neurons=3, hidden_neurons=3, output_neurons=1)

        learning_rate_bp = StringVar()

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

            label_lr_bp = ctk.CTkLabel(tab, text="Learning rate")
            label_lr_bp.grid(row=1, column=0, padx=(20, 0), pady=(20, 20))

            # Use learning_rate as the text variable for the entry widget
            entry_lr_bp = ctk.CTkEntry(tab, textvariable=learning_rate_bp)
            entry_lr_bp.grid(row=1, column=1, padx=(20, 0), pady=(20, 20))

            # Attach the trace callback to the text variable
            learning_rate_bp.trace("w", parse_entry)

        entry_learning_rate_bp()

        def data_options():
            # Images for the labels
            pil_xor = Image.open(
                r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\backprop\data\xor_gate.jpeg")
            resized_image_xor = pil_xor.resize((120, 120))
            image_xor = ImageTk.PhotoImage(resized_image_xor)

            pil_xnor = Image.open(
                r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\backprop\data\xnor_gate.jpeg")
            resized_image_xnor = pil_xnor.resize((120, 120))
            image_xnor = ImageTk.PhotoImage(resized_image_xnor)

            pil_nand = Image.open(
                r"\Users\cvill\OneDrive\Documents\GitHub\Advanced-Neural-Network-ToolKit\neural_networks\images\backprop\data\nand_gate.jpeg")
            resized_image_nand = pil_nand.resize((120, 120))
            image_nand = ImageTk.PhotoImage(resized_image_nand)

            # select the image that you want to decode, Variable to store the image of choice
            self.dataset_choice_var = ctk.StringVar(value="XOR")  # Default is "First Image"

            dataset_choice_label = ctk.CTkLabel(tab, text="Dataset:", font=("bold", 24))
            dataset_choice_label.grid(row=2, column=0, padx=(20, 0), pady=(20, 20))

            # Radio buttons for the selecting datasets
            xor_radio = ctk.CTkRadioButton(tab, text="XOR", variable=self.dataset_choice_var, value="XOR")
            xor_radio.grid(row=3, column=0, padx=(20, 0), pady=(5, 5))
            xor_label = ctk.CTkLabel(tab, image=image_xor, text="")
            xor_label.grid(row=4, column=0, padx=(20, 0), pady=(5, 5))

            xnor_radio = ctk.CTkRadioButton(tab, text="XNOR", variable=self.dataset_choice_var, value="XNOR")
            xnor_radio.grid(row=3, column=1, padx=(20, 0), pady=(5, 5), sticky="w")
            xnor_label = ctk.CTkLabel(tab, image=image_xnor, text="")
            xnor_label.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

            nand_radio = ctk.CTkRadioButton(tab, text="NAND", variable=self.dataset_choice_var, value="NAND")
            nand_radio.grid(row=3, column=2, padx=(20, 0), pady=(5, 5), sticky="w")
            nand_label = ctk.CTkLabel(tab, image=image_nand, text="")
            nand_label.grid(row=4, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

            self.xor_data_array = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
            self.xor_labels_array = np.array([[0], [1], [1], [0]])

            self.xnor_data_array = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
            self.xnor_labels_array = np.array([[1], [0], [0], [1]])

            self.nand_data_array = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
            self.nand_labels_array = np.array([[1], [1], [1], [0]])

        data_options()

        # ######## Function to obtain the chosen dataset ########
        def get_selected_dataset():
            selected_value = self.dataset_choice_var.get()
            if selected_value == "XOR":
                return self.xor_data_array, self.xor_labels_array
            elif selected_value == "XNOR":
                return self.xnor_data_array, self.xnor_labels_array
            elif selected_value == "NAND":
                return self.nand_data_array, self.nand_labels_array
            else:
                return None, None

        # train the network
        def train_backprop():
            data, label = get_selected_dataset()
            if data is not None and label is not None:
                print("Train phase")
                learning_rate = float(learning_rate_bp.get())
                backpropagation_model.train(data, label, learning_rate)
            else:
                print("An error occurred")

        self.label_results = ctk.CTkLabel(tab, text="", wraplength=300)
        self.label_results.grid(row=6, column=0, padx=20, pady=20, sticky="ew")

        # test the network
        def test_backprop():
            data, label = get_selected_dataset()
            if data is not None and label is not None:
                print("Test phase")
                _, _, results_string = backpropagation_model.test(data, label)
                self.label_results.configure(text=results_string)
            else:
                print("An error occurred")

        # Plot loss
        def plot_loss_from_test():
            # Plot the loss directly using the loss values obtained during the test
            loss_values = backpropagation_model.loss

            with plt.style.context('seaborn-darkgrid'):
                plt.figure(figsize=(6, 4))
                plt.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Mean Square Error')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title("Training Loss")
                plt.legend()
                plt.grid(True)
                plt.show()

        button_train_bp = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                  text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                  command=train_backprop)

        button_train_bp.grid(row=5, column=0, padx=(20, 20), pady=(20, 20)) # , sticky="nsew"

        button_test_bp = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                 text="Test Network", text_color=("gray10", "#DCE4EE"),
                                                 command=test_backprop)
        button_test_bp.grid(row=5, column=1, padx=(20, 20), pady=(20, 20))

        button_loss_bp = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                 text="Plot loss", text_color=("gray10", "#DCE4EE"),
                                                 command=plot_loss_from_test)
        button_loss_bp.grid(row=5, column=2, padx=(20, 20), pady=(20, 20))

    # ########### 3rd Tab ###########
    def setup_som_tab(self, tab):
        label_tab_3 = ctk.CTkLabel(tab, text="Kohonen SOM Network", font=("bold", 24))
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

            label_num_points_pclass = ctk.CTkLabel(tab, text="Num. of points per class:")
            label_num_points_pclass.grid(row=1, column=0, padx=20, pady=20)

            entry_num_points_per_class_som = ctk.CTkEntry(tab, placeholder_text="Number of points per class",
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

            label_num_classes_som = ctk.CTkLabel(tab, text="Num. of classes:")
            label_num_classes_som.grid(row=1, column=2, padx=20, pady=20)

            entry_num_classes_som = ctk.CTkEntry(tab, textvariable=num_classes_som)
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

            label_number_of_neurons_som = ctk.CTkLabel(tab, text="Num of neurons:")
            label_number_of_neurons_som.grid(row=2, column=0, padx=20, pady=20)

            entry_number_of_neurons_som = ctk.CTkEntry(tab, textvariable=num_of_neurons_som)
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

            label_input_dimension = ctk.CTkLabel(tab, text="Input dimension:")
            label_input_dimension.grid(row=2, column=2, padx=20, pady=20)

            entry_input_dimension_som = ctk.CTkEntry(tab, textvariable=input_dim_som)
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

            label_learning_rate_som = ctk.CTkLabel(tab, text="Learning rate")
            label_learning_rate_som.grid(row=3, column=0, padx=20, pady=20)

            entry_learning_rate_som = ctk.CTkEntry(tab, textvariable=lr_som)
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

            label_epoch_max_som = ctk.CTkLabel(tab, text="Epoch max:")
            label_epoch_max_som.grid(row=3, column=2, padx=20, pady=20)

            entry_epoch_max_som = ctk.CTkEntry(tab, textvariable=epoch_max_som)
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

        label_rand_data = ctk.CTkLabel(tab, text="Random Data", font=("bold", 18))
        label_rand_data.grid(row=2, column=4, padx=(20, 0), pady=(5, 5), sticky="w")

        image_label_rand_data = ctk.CTkLabel(tab, image=image_rand_data, text="")
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

                # Now, you can use the data to train your neural network using the SOM model
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

        button_generate_data_som = ctk.CTkButton(tab, fg_color="#219ebc", border_width=2,
                                                           text="Generate data", text_color="white",
                                                           command=generate_data)
        # command=generate_data_som()) num_points_p_class=, num_classes=
        button_generate_data_som.grid(row=7, column=0, padx=(20, 20), pady=(20, 20))

        button_train_network = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                       text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                       command=train_som)
        button_train_network.grid(row=7, column=1, padx=(20, 20), pady=(20, 20))

        button_plot_results = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                      text="Plot results", text_color=("gray10", "#DCE4EE"),
                                                      command=plot_som)
        button_plot_results.grid(row=7, column=2, padx=(20, 20), pady=(20, 20))

    # ########### 4th Tab ###########
    def setup_autoencoder(self, tab):

        learning_rate_ac = StringVar()
        momentum_ac = StringVar()
        epoch_max_ac = StringVar()

        label_tab = ctk.CTkLabel(tab, text="AutoEncoder Network", font=("bold", 24))
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

            label_learning_rate = ctk.CTkLabel(tab, text="Learning rate")
            label_learning_rate.grid(row=1, column=0, padx=(20, 0), pady=(20, 20))

            entry_learning_rate = ctk.CTkEntry(tab, textvariable=learning_rate_ac)
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

            label_momentum_rate = ctk.CTkLabel(tab, text="Momentum rate")
            label_momentum_rate.grid(row=1, column=2, padx=(20, 0), pady=(20, 20))  # , sticky="nsew"

            entry_momentum_rate = ctk.CTkEntry(tab, textvariable=momentum_ac)
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

            label_epoch_max = ctk.CTkLabel(tab, text="Epoch max")
            label_epoch_max.grid(row=2, column=0, padx=(20, 0), pady=(20, 20))

            entry_epoch_max = ctk.CTkEntry(tab, textvariable=epoch_max_ac)
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
        image_choice_var = ctk.StringVar(value="First Image")  # Default is "First Image"

        # Radio buttons for selecting the Images
        first_image_radio = ctk.CTkRadioButton(tab, text="First Image",
                                                         variable=image_choice_var, value="First Image")
        first_image_radio.grid(row=3, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        first_image_label = ctk.CTkLabel(tab, image=image1, text="")
        first_image_label.grid(row=4, column=0, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_radio = ctk.CTkRadioButton(tab, text="Second Image",
                                                          variable=image_choice_var, value="Second Image")
        second_image_radio.grid(row=3, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        second_image_label = ctk.CTkLabel(tab, image=image2, text="")
        second_image_label.grid(row=4, column=1, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_radio = ctk.CTkRadioButton(tab, text="Third Image",
                                                         variable=image_choice_var, value="Third Image")
        third_image_radio.grid(row=3, column=2, padx=(20, 0), pady=(5, 5), sticky="w")

        third_image_label = ctk.CTkLabel(tab, image=image3, text="")
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
        img_1_array = np.array(img_1)  # / 255.0 * 2 - 1
        img_2_array = np.array(img_2)  # / 255.0 * 2 - 1
        img_3_array = np.array(img_3)  # / 255.0 * 2 - 1

        # ######## Function to obtain the chosen image ########
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

                self.loss, latent_space, self.decoded_inputs = autoencoder_model.train(data, learning_rate_ac_value,
                                                                                       momentum_ac_value,
                                                                                       epoch_max_ac_value)
            else:
                print("The model need's the hyperparameters")

        # Modify the function call to plot_images
        def reconstruct_selected_image():
            data = get_selected_image_data()
            if data is not None:
                reconstructed_image = self.decoded_inputs.reshape(16, 16,
                                                                  3)  # This should be a NumPy array. If it's not, you'll have to adjust this part.
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

        button_train_ac = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                  text="Train Network", text_color=("gray10", "#DCE4EE"),
                                                  command=train_ac)
        button_train_ac.grid(row=6, column=0, padx=(20, 20), pady=(20, 20))

        button_reconstruction_ac = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                           text="Reconstruct image",
                                                           text_color=("gray10", "#DCE4EE"),
                                                           anchor="w", command=reconstruct_selected_image)
        button_reconstruction_ac.grid(row=6, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")

        button_plot_loss_ac = ctk.CTkButton(tab, fg_color="transparent", border_width=2,
                                                      text="Plot loss", text_color=("gray10", "#DCE4EE"),
                                                      command=plot_loss_from_training)
        button_plot_loss_ac.grid(row=6, column=2, padx=(20, 20), pady=(20, 20))

    def plot_images(self, original_image, reconstructed_img):
        """Plot the original, noisy, and reconstructed images."""
        plt.figure(figsize=(8, 4))

        # Reshape and convert the original image to a NumPy array
        original_image_array = np.array(original_image[0])
        original_image_array = original_image_array.astype(np.uint8)  # + 1) / 2 * 255

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
        label_lvq = ctk.CTkLabel(tab, text="LVQ Network", font=("bold", 24))
        label_lvq.grid(row=0, column=0, padx=20, pady=20)

        input_lvq = StringVar()
        labels_lvq = StringVar()
        learning_rate_lvq = StringVar()
        epoch_max_lvq = StringVar()

    def exit(self):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
