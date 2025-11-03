# dataloaders.py

import os
import re
import glob
import logging
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GraphDataset(Dataset):
    def __init__(self, initial_graph_dir, final_graph_dir, settings_dir, task='predict_n6d',
                 use_edge_attr=False, edge_attr_method="v1", preload_data=False,
                 expected_initial_x_dim=6, expected_settings_dim=6):
        """
        Initializes the GraphDataset.

        Args:
            initial_graph_dir (str): Directory containing initial graph .pt files.
            final_graph_dir (str): Directory containing final graph .pt files.
            settings_dir (str): Directory containing settings .pt files.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
                                  Defaults to 'predict_n6d'.
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
                                            Defaults to False.
            edge_attr_method (str, optional): Method for edge attribute computation.
                                              Defaults to 'v0'.
            preload_data (bool, optional): If True, preloads all data into memory.
                                           Defaults to False.
            expected_initial_x_dim (int, optional): Expected dimension of initial node features. Defaults to 6.
            expected_settings_dim (int, optional): Expected dimension of settings features. Defaults to 6.
        """
        self.initial_graph_dir = initial_graph_dir
        self.final_graph_dir = final_graph_dir
        self.settings_dir = settings_dir
        self.task = task
        self.use_edge_attr = use_edge_attr
        self.edge_attr_method = edge_attr_method
        self.preload_data = preload_data
        self.expected_initial_x_dim = expected_initial_x_dim
        self.expected_settings_dim = expected_settings_dim
        self.expected_total_x_dim = self.expected_initial_x_dim + self.expected_settings_dim

        # Load and sort initial, final, and settings files
        self.initial_files = sorted(
            [f for f in os.listdir(initial_graph_dir) if f.endswith('.pt')],
            key=lambda f: int(re.search(r'graph_(\d+)', f).group(1))
        )
        self.final_files = sorted(
            [f for f in os.listdir(final_graph_dir) if f.endswith('.pt')],
            key=lambda f: int(re.search(r'graph_(\d+)', f).group(1))
        )
        self.settings_files = sorted(
            [f for f in os.listdir(settings_dir) if f.endswith('.pt')],
            key=lambda f: int(re.search(r'settings_(\d+)', f).group(1))
        )

        assert len(self.initial_files) == len(self.final_files) == len(self.settings_files), \
            "Mismatch in number of initial graphs, final graphs, and settings files."

        # Preload data if enabled
        if self.preload_data:
            logging.info("Preloading and validating all data samples...")
            self.initial_graphs = []
            self.final_graphs = []
            self.settings = []
            for idx, (init_file, final_file, settings_file) in enumerate(zip(self.initial_files, self.final_files, self.settings_files)):
                try:
                    initial_data = torch.load(os.path.join(initial_graph_dir, init_file), weights_only=False)
                    final_data = torch.load(os.path.join(final_graph_dir, final_file), weights_only=False)
                    setting = torch.load(os.path.join(settings_dir, settings_file), weights_only=False)

                    # **Step 1: Verify Initial Node Feature Dimension**
                    if initial_data.x.shape[1] != self.expected_initial_x_dim:
                        logging.error(f"Sample {idx}: Expected initial node feature dimension {self.expected_initial_x_dim}, "
                                      f"got {initial_data.x.shape[1]}.")
                        raise ValueError(f"Sample {idx}: Inconsistent initial node feature dimensions.")

                    # **Step 2: Verify Settings Dimension**
                    if setting.shape[0] != self.expected_settings_dim:
                        logging.error(f"Sample {idx}: Expected settings dimension {self.expected_settings_dim}, "
                                      f"got {setting.shape[0]}.")
                        raise ValueError(f"Sample {idx}: Inconsistent settings dimensions.")

                    # **Step 3: Concatenate Settings to Node Features**
                    num_nodes = initial_data.num_nodes
                    settings_expanded = setting.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, D_settings]
                    initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # Shape: [num_nodes, D_total]

                    # **Step 4: Verify Total Node Feature Dimension After Concatenation**
                    if initial_data.x.shape[1] != self.expected_total_x_dim:
                        logging.error(f"Sample {idx}: Expected total node feature dimension {self.expected_total_x_dim}, "
                                      f"got {initial_data.x.shape[1]}.")
                        raise ValueError(f"Sample {idx}: Inconsistent total node feature dimensions after concatenation.")

                    # **Step 5: Extract Positions**
                    initial_data.pos = initial_data.x[:, :3]  # Assuming first 3 features are x, y, z

                    # **Step 6: Compute Edge Attributes if Required**
                    if self.use_edge_attr:
                        self._compute_edge_attr(initial_data, idx)
                    else:
                        initial_data.edge_attr = None  # Explicitly set to None if not used
                        logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

                    # **Step 7: Assign Target Node Features Based on Task**
                    if self.task == 'predict_n6d':
                        initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
                    elif self.task == 'predict_n4d':
                        initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
                    elif self.task == 'predict_n2d':
                        initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
                    else:
                        raise ValueError(f"Unknown task: {self.task}")

                    # **Step 8: Initialize 'batch' Attribute if Missing**
                    if not hasattr(initial_data, 'batch') or initial_data.batch is None:
                        initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)
                        logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

                    # Append preloaded and validated data
                    self.initial_graphs.append(initial_data)
                    self.final_graphs.append(final_data)
                    self.settings.append(setting)

                except Exception as e:
                    logging.error(f"Error processing sample {idx}: {e}")
                    raise e

            logging.info(f"Preloaded and validated {len(self.initial_graphs)} samples successfully.")
        else:
            self.initial_graphs = None
            self.final_graphs = None
            self.settings = None

        logging.info(f"Initialized GraphDataset with {len(self)} samples. "
                     f"Preload data: {self.preload_data}, Use edge_attr: {self.use_edge_attr}, "
                     f"Edge attr method: {self.edge_attr_method}")

    def __len__(self):
        return len(self.initial_files)

    def __getitem__(self, idx):
        """
        Retrieves the graph data at the specified index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            torch_geometric.data.Data: Graph data object containing node features, edge indices,
                                       edge attributes (if required), target labels, positions,
                                       and batch information.
        """
        if self.preload_data:
            initial_data = self.initial_graphs[idx]
            final_data = self.final_graphs[idx]
            setting = self.settings[idx]
        else:
            initial_data = torch.load(os.path.join(self.initial_graph_dir, self.initial_files[idx]), weights_only=False)
            final_data = torch.load(os.path.join(self.final_graph_dir, self.final_files[idx]), weights_only=False)
            setting = torch.load(os.path.join(self.settings_dir, self.settings_files[idx]), weights_only=False)

            # **Step 1: Verify Individual Feature Dimensions Before Concatenation**
            # Verify initial node feature dimension
            if initial_data.x.shape[1] != self.expected_initial_x_dim:
                logging.error(f"Sample {idx}: Expected initial node feature dimension {self.expected_initial_x_dim}, "
                              f"got {initial_data.x.shape[1]}.")
                raise ValueError(f"Sample {idx}: Inconsistent initial node feature dimensions.")

            # Verify settings feature dimension
            if setting.shape[0] != self.expected_settings_dim:
                logging.error(f"Sample {idx}: Expected settings dimension {self.expected_settings_dim}, "
                              f"got {setting.shape[0]}.")
                raise ValueError(f"Sample {idx}: Inconsistent settings dimensions.")

            # **Step 2: Concatenate Settings to Node Features**
            num_nodes = initial_data.num_nodes
            settings_expanded = setting.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, D_settings]
            initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # Shape: [num_nodes, D_total]

            # **Step 3: Verify Total Node Feature Dimension After Concatenation**
            if initial_data.x.shape[1] != self.expected_total_x_dim:
                logging.error(f"Sample {idx}: Expected total node feature dimension {self.expected_total_x_dim}, "
                              f"got {initial_data.x.shape[1]}.")
                raise ValueError(f"Sample {idx}: Inconsistent total node feature dimensions after concatenation.")

            # **Step 4: Extract Positions**
            initial_data.pos = initial_data.x[:, :3]  # Assuming first 3 features are x, y, z

            # **Step 5: Compute Edge Attributes if Required**
            if self.use_edge_attr:
                self._compute_edge_attr(initial_data, idx)
            else:
                initial_data.edge_attr = None  # Explicitly set to None if not used
                logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

            # **Step 6: Assign Target Node Features Based on Task**
            if self.task == 'predict_n6d':
                initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
            elif self.task == 'predict_n4d':
                initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
            elif self.task == 'predict_n2d':
                initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
            else:
                raise ValueError(f"Unknown task: {self.task}")

            # **Step 7: Initialize 'batch' Attribute if Missing**
            if not hasattr(initial_data, 'batch') or initial_data.batch is None:
                initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)
                logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

        return initial_data

    def _compute_edge_attr(self, graph, idx):
        """
        Computes and assigns edge attributes based on the selected method.

        Args:
            graph (torch_geometric.data.Data): The graph data object.
            idx (int): Index of the data sample.
        """
        row, col = graph.edge_index
        pos_diff = graph.pos[row] - graph.pos[col]  # Shape: [num_edges, 3]

        if self.edge_attr_method == "v0":
            # Use pos_diff only, with standardization
            edge_attr = pos_diff  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v0n":
            # Use normalized pos_diff, without standardization
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            edge_attr = pos_diff / norm  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v1":
            # Use pos_diff + distance, all standardized
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v1n":
            # Use normalized pos_diff + distance, only distance standardized
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            normalized_pos_diff = pos_diff / norm  # Shape: [num_edges, 3]
            edge_attr = torch.cat([normalized_pos_diff, norm], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2":
            # Use pos_diff + 1/squared distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2n":
            # Use normalized pos_diff + 1/squared distance, only 1/squared distance standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            normalized_pos_diff = pos_diff / torch.sqrt(squared_distance)  # Shape: [num_edges, 3]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([normalized_pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v3":
            # Use pos_diff / squared_distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            edge_attr = pos_diff / squared_distance  # Shape: [num_edges, 3]
        else:
            raise ValueError(f"Invalid edge_attr_method: {self.edge_attr_method}")

        # Standardization logic
        if self.edge_attr_method in ["v0", "v1", "v2", "v3"]:
            # Standardize all features
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, ...]
        elif self.edge_attr_method in ["v1n", "v2n"]:
            # Standardize only the magnitude-related feature (last column)
            eps = 1e-10
            magnitude_indices = [-1]  # Last column contains the magnitude feature
            edge_attr_mean = edge_attr[:, magnitude_indices].mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr[:, magnitude_indices].std(dim=0, keepdim=True)
            edge_attr[:, magnitude_indices] = (edge_attr[:, magnitude_indices] - edge_attr_mean) / (edge_attr_std + eps)

        graph.edge_attr = edge_attr
        logging.debug(f"Sample {idx}: Computed edge_attr with shape {graph.edge_attr.shape} using method {self.edge_attr_method}")


class StepPairGraphDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step, final_step, task='predict_n6d', use_settings=False,
                 identical_settings=False, settings_file=None, use_edge_attr=False, edge_attr_method="v0",
                 preload_data=False, subsample_size=None,
                 expected_initial_x_dim=6, expected_settings_dim=6):
        """
        Initializes the StepPairGraphDataset.

        Args:
            graph_data_dir (str): Base directory containing the graph data organized by sequence steps.
            initial_step (int): Index of the initial sequence step.
            final_step (int): Index of the final sequence step.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
            use_settings (bool, optional): Flag indicating whether to use settings.
            identical_settings (bool, optional): Whether settings are identical across samples.
            settings_file (str, optional): Path to the settings file (used if identical_settings is True).
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
            edge_attr_method (str, optional): Method for edge attribute computation.
            preload_data (bool, optional): If True, preloads all data into memory.
            subsample_size (int, optional): Number of samples to use from the dataset. Use all data if not specified.
            expected_initial_x_dim (int, optional): Expected dimension of initial node features. Defaults to 6.
            expected_settings_dim (int, optional): Expected dimension of settings features. Defaults to 6.
        """
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.task = task
        self.use_settings = use_settings
        self.identical_settings = identical_settings
        self.settings_file = settings_file
        self.use_edge_attr = use_edge_attr
        self.edge_attr_method = edge_attr_method
        self.preload_data = preload_data
        self.subsample_size = subsample_size
        self.expected_initial_x_dim = expected_initial_x_dim
        self.expected_settings_dim = expected_settings_dim
        self.expected_total_x_dim = self.expected_initial_x_dim + self.expected_settings_dim

        # Build file paths for initial and final graphs
        initial_graph_dir = os.path.join(graph_data_dir, f"step_{initial_step}")
        final_graph_dir = os.path.join(graph_data_dir, f"step_{final_step}")

        # Function to extract the graph number from filenames
        def extract_graph_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'graph_(\d+)\.pt', filename)
            if match:
                return int(match.group(1))
            else:
                raise ValueError(f"Filename {filename} does not match pattern 'graph_X.pt'.")

        # Get list of initial and final graph files and sort them by graph number
        self.initial_graph_files = sorted(
            glob.glob(os.path.join(initial_graph_dir, 'graph_*.pt')),
            key=extract_graph_number
        )
        self.final_graph_files = sorted(
            glob.glob(os.path.join(final_graph_dir, 'graph_*.pt')),
            key=extract_graph_number
        )

        # Subsample the dataset if subsample_size is specified
        if self.subsample_size is not None:
            self.initial_graph_files = self.initial_graph_files[:self.subsample_size]
            self.final_graph_files = self.final_graph_files[:self.subsample_size]

        if len(self.initial_graph_files) != len(self.final_graph_files):
            raise ValueError("Mismatch in number of initial and final graph files.")

        # Load settings if identical
        if self.use_settings:
            if self.identical_settings:
                if settings_file is None:
                    raise ValueError("Settings file must be provided when identical_settings is True.")
                self.settings = torch.load(settings_file, weights_only=False)
            else:
                # Load settings per sample if necessary
                self.settings_files = [f.replace(f"step_{initial_step}", "settings").replace('graph_', 'settings_')
                                       for f in self.initial_graph_files]
                if self.subsample_size is not None:
                    self.settings_files = self.settings_files[:self.subsample_size]
                if not all(os.path.isfile(f) for f in self.settings_files):
                    raise ValueError("Some settings files are missing.")

        # Preload data if enabled
        if self.preload_data:
            logging.info("Preloading and validating all data samples...")
            self.initial_graphs = []
            self.final_graphs = []
            self.settings = []
            for idx, (init_file, final_file) in enumerate(zip(self.initial_graph_files, self.final_graph_files)):
                try:
                    initial_data = torch.load(init_file, weights_only=False)
                    final_data = torch.load(final_file, weights_only=False)
                    if self.use_settings and not self.identical_settings:
                        setting = torch.load(self.settings_files[idx], weights_only=False)
                    elif self.use_settings and self.identical_settings:
                        setting = self.settings
                    else:
                        setting = None

                    # **Step 1: Verify Initial Node Feature Dimension**
                    if initial_data.x.shape[1] != self.expected_initial_x_dim:
                        logging.error(f"Sample {idx}: Expected initial node feature dimension {self.expected_initial_x_dim}, "
                                      f"got {initial_data.x.shape[1]}.")
                        raise ValueError(f"Sample {idx}: Inconsistent initial node feature dimensions.")

                    # **Step 2: Verify Settings Dimension**
                    if self.use_settings:
                        if setting is None:
                            logging.error(f"Sample {idx}: Settings are enabled but settings data is missing.")
                            raise ValueError(f"Sample {idx}: Missing settings data.")
                        if setting.shape[0] != self.expected_settings_dim:
                            logging.error(f"Sample {idx}: Expected settings dimension {self.expected_settings_dim}, "
                                          f"got {setting.shape[0]}.")
                            raise ValueError(f"Sample {idx}: Inconsistent settings dimensions.")

                    # **Step 3: Concatenate Settings to Node Features**
                    if self.use_settings:
                        num_nodes = initial_data.num_nodes
                        settings_expanded = setting.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, D_settings]
                        initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # Shape: [num_nodes, D_total]

                        # **Step 4: Verify Total Node Feature Dimension After Concatenation**
                        if initial_data.x.shape[1] != self.expected_total_x_dim:
                            logging.error(f"Sample {idx}: Expected total node feature dimension {self.expected_total_x_dim}, "
                                          f"got {initial_data.x.shape[1]}.")
                            raise ValueError(f"Sample {idx}: Inconsistent total node feature dimensions after concatenation.")

                    # **Step 5: Extract Positions**
                    initial_data.pos = initial_data.x[:, :3]  # Assuming first 3 features are x, y, z

                    # **Step 6: Compute Edge Attributes if Required**
                    if self.use_edge_attr:
                        self._compute_edge_attr(initial_data, idx)
                    else:
                        initial_data.edge_attr = None  # Explicitly set to None if not used
                        logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

                    # **Step 7: Assign Target Node Features Based on Task**
                    if self.task == 'predict_n6d':
                        initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
                    elif self.task == 'predict_n4d':
                        initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
                    elif self.task == 'predict_n2d':
                        initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
                    else:
                        raise ValueError(f"Unknown task: {self.task}")

                    # **Step 8: Initialize 'batch' Attribute if Missing**
                    if not hasattr(initial_data, 'batch') or initial_data.batch is None:
                        initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)
                        logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

                    # Append preloaded and validated data
                    self.initial_graphs.append(initial_data)
                    self.final_graphs.append(final_data)
                    if self.use_settings and not self.identical_settings:
                        self.settings.append(setting)

                except Exception as e:
                    logging.error(f"Error processing sample {idx}: {e}")
                    raise e

            logging.info(f"Preloaded and validated {len(self.initial_graphs)} samples successfully.")
        else:
            self.initial_graphs = None
            self.final_graphs = None
            if self.use_settings and not self.identical_settings:
                self.settings = None

        logging.info(f"Initialized StepPairGraphDataset with {len(self)} samples. "
                     f"Preload data: {self.preload_data}, Use edge_attr: {self.use_edge_attr}, "
                     f"Edge attr method: {self.edge_attr_method}")

    def __len__(self):
        return len(self.initial_graph_files)

    def __getitem__(self, idx):
        if self.preload_data:
            initial_graph = self.initial_graphs[idx]
            final_graph = self.final_graphs[idx]
            if self.use_settings and not self.identical_settings:
                setting = self.settings[idx]
            elif self.use_settings and self.identical_settings:
                setting = self.settings
            else:
                setting = None
        else:
            initial_graph = torch.load(self.initial_graph_files[idx], weights_only=False)
            final_graph = torch.load(self.final_graph_files[idx], weights_only=False)
            if self.use_settings and not self.identical_settings:
                setting = torch.load(self.settings_files[idx], weights_only=False)
            elif self.use_settings and self.identical_settings:
                setting = self.settings
            else:
                setting = None

            # **Step 1: Verify Individual Feature Dimensions Before Concatenation**
            # Verify initial node feature dimension
            if initial_graph.x.shape[1] != self.expected_initial_x_dim:
                logging.error(f"Sample {idx}: Expected initial node feature dimension {self.expected_initial_x_dim}, "
                              f"got {initial_graph.x.shape[1]}.")
                raise ValueError(f"Sample {idx}: Inconsistent initial node feature dimensions.")

            # Verify settings feature dimension
            if self.use_settings:
                if setting is None:
                    logging.error(f"Sample {idx}: Settings are enabled but settings data is missing.")
                    raise ValueError(f"Sample {idx}: Missing settings data.")
                if setting.shape[0] != self.expected_settings_dim:
                    logging.error(f"Sample {idx}: Expected settings dimension {self.expected_settings_dim}, "
                                  f"got {setting.shape[0]}.")
                    raise ValueError(f"Sample {idx}: Inconsistent settings dimensions.")

            # **Step 2: Concatenate Settings to Node Features (If Applicable)**
            if self.use_settings:
                num_nodes = initial_graph.num_nodes
                settings_expanded = setting.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, D_settings]
                initial_graph.x = torch.cat([initial_graph.x, settings_expanded], dim=1)  # Shape: [num_nodes, D_total]

                # **Step 3: Verify Total Node Feature Dimension After Concatenation**
                if initial_graph.x.shape[1] != self.expected_total_x_dim:
                    logging.error(f"Sample {idx}: Expected total node feature dimension {self.expected_total_x_dim}, "
                                  f"got {initial_graph.x.shape[1]}.")
                    raise ValueError(f"Sample {idx}: Inconsistent total node feature dimensions after concatenation.")

            # **Optional: Log shapes for the first few samples**
            if idx < 5 and self.use_settings:  # Adjust the number as needed
                logging.debug(f"Sample {idx}: Node features shape after concatenation: {initial_graph.x.shape}")

            # **Step 4: Extract Positions**
            initial_graph.pos = initial_graph.x[:, :3]  # Assuming first 3 features are x, y, z coordinates

            # **Step 5: Compute Edge Attributes if Required**
            if self.use_edge_attr:
                self._compute_edge_attr(initial_graph, idx)
            else:
                initial_graph.edge_attr = None  # Explicitly set to None if not used
                logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

            # **Step 6: Assign Target Node Features Based on Task**
            if self.task == 'predict_n6d':
                initial_graph.y = final_graph.x[:, :6]  # Shape: [num_nodes, 6]
            elif self.task == 'predict_n4d':
                initial_graph.y = final_graph.x[:, :4]  # Shape: [num_nodes, 4]
            elif self.task == 'predict_n2d':
                initial_graph.y = final_graph.x[:, :2]  # Shape: [num_nodes, 2]
            else:
                raise ValueError(f"Unknown task: {self.task}")

            # **Step 7: Initialize 'batch' Attribute if Missing**
            if not hasattr(initial_graph, 'batch') or initial_graph.batch is None:
                initial_graph.batch = torch.zeros(initial_graph.num_nodes, dtype=torch.long)
                logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

        return initial_graph

    def _compute_edge_attr(self, graph, idx):
        """
        Computes and assigns edge attributes based on the selected method.

        Args:
            graph (torch_geometric.data.Data): The graph data object.
            idx (int): Index of the data sample.
        """
        row, col = graph.edge_index
        pos_diff = graph.pos[row] - graph.pos[col]  # Shape: [num_edges, 3]

        if self.edge_attr_method == "v0":
            # Use pos_diff only, with standardization
            edge_attr = pos_diff  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v0n":
            # Use normalized pos_diff, without standardization
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            edge_attr = pos_diff / norm  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v1":
            # Use pos_diff + distance, all standardized
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v1n":
            # Use normalized pos_diff + distance, only distance standardized
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            normalized_pos_diff = pos_diff / norm  # Shape: [num_edges, 3]
            edge_attr = torch.cat([normalized_pos_diff, norm], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2":
            # Use pos_diff + 1/squared distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2n":
            # Use normalized pos_diff + 1/squared distance, only 1/squared distance standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            normalized_pos_diff = pos_diff / torch.sqrt(squared_distance)  # Shape: [num_edges, 3]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([normalized_pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v3":
            # Use pos_diff / squared_distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            edge_attr = pos_diff / squared_distance  # Shape: [num_edges, 3]
        else:
            raise ValueError(f"Invalid edge_attr_method: {self.edge_attr_method}")

        # Standardization logic
        if self.edge_attr_method in ["v0", "v1", "v2", "v3"]:
            # Standardize all features
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, ...]
        elif self.edge_attr_method in ["v1n", "v2n"]:
            # Standardize only the magnitude-related feature (last column)
            eps = 1e-10
            magnitude_indices = [-1]  # Last column contains the magnitude feature
            edge_attr_mean = edge_attr[:, magnitude_indices].mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr[:, magnitude_indices].std(dim=0, keepdim=True)
            edge_attr[:, magnitude_indices] = (edge_attr[:, magnitude_indices] - edge_attr_mean) / (edge_attr_std + eps)

        graph.edge_attr = edge_attr
        logging.debug(f"Sample {idx}: Computed edge_attr with shape {graph.edge_attr.shape} using method {self.edge_attr_method}")


class StepPairGraphDataset(Dataset):
    def __init__(self, graph_data_dir, initial_step, final_step, task='predict_n6d', use_settings=False,
                 identical_settings=False, settings_file=None, use_edge_attr=False, edge_attr_method="v0",
                 preload_data=False, subsample_size=None,
                 expected_initial_x_dim=6, expected_settings_dim=6):
        """
        Initializes the StepPairGraphDataset.

        Args:
            graph_data_dir (str): Base directory containing the graph data organized by sequence steps.
            initial_step (int): Index of the initial sequence step.
            final_step (int): Index of the final sequence step.
            task (str, optional): Prediction task. One of ['predict_n6d', 'predict_n4d', 'predict_n2d'].
            use_settings (bool, optional): Flag indicating whether to use settings.
            identical_settings (bool, optional): Whether settings are identical across samples.
            settings_file (str, optional): Path to the settings file (used if identical_settings is True).
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
            edge_attr_method (str, optional): Method for edge attribute computation.
            preload_data (bool, optional): If True, preloads all data into memory.
            subsample_size (int, optional): Number of samples to use from the dataset. Use all data if not specified.
            expected_initial_x_dim (int, optional): Expected dimension of initial node features. Defaults to 6.
            expected_settings_dim (int, optional): Expected dimension of settings features. Defaults to 6.
        """
        self.graph_data_dir = graph_data_dir
        self.initial_step = initial_step
        self.final_step = final_step
        self.task = task
        self.use_settings = use_settings
        self.identical_settings = identical_settings
        self.settings_file = settings_file
        self.use_edge_attr = use_edge_attr
        self.edge_attr_method = edge_attr_method
        self.preload_data = preload_data
        self.subsample_size = subsample_size
        self.expected_initial_x_dim = expected_initial_x_dim
        self.expected_settings_dim = expected_settings_dim
        self.expected_total_x_dim = self.expected_initial_x_dim + self.expected_settings_dim

        # Build file paths for initial and final graphs
        initial_graph_dir = os.path.join(graph_data_dir, f"step_{initial_step}")
        final_graph_dir = os.path.join(graph_data_dir, f"step_{final_step}")

        # Function to extract the graph number from filenames
        def extract_graph_number(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'graph_(\d+)\.pt', filename)
            if match:
                return int(match.group(1))
            else:
                raise ValueError(f"Filename {filename} does not match pattern 'graph_X.pt'.")

        # Get list of initial and final graph files and sort them by graph number
        self.initial_graph_files = sorted(
            glob.glob(os.path.join(initial_graph_dir, 'graph_*.pt')),
            key=extract_graph_number
        )
        self.final_graph_files = sorted(
            glob.glob(os.path.join(final_graph_dir, 'graph_*.pt')),
            key=extract_graph_number
        )

        # Subsample the dataset if subsample_size is specified
        if self.subsample_size is not None:
            self.initial_graph_files = self.initial_graph_files[:self.subsample_size]
            self.final_graph_files = self.final_graph_files[:self.subsample_size]

        if len(self.initial_graph_files) != len(self.final_graph_files):
            raise ValueError("Mismatch in number of initial and final graph files.")

        # Load settings if identical
        if self.use_settings:
            if self.identical_settings:
                if settings_file is None:
                    raise ValueError("Settings file must be provided when identical_settings is True.")
                self.settings = torch.load(settings_file, weights_only=False)
            else:
                # Load settings per sample if necessary
                self.settings_files = [f.replace(f"step_{initial_step}", "settings").replace('graph_', 'settings_')
                                       for f in self.initial_graph_files]
                if self.subsample_size is not None:
                    self.settings_files = self.settings_files[:self.subsample_size]
                if not all(os.path.isfile(f) for f in self.settings_files):
                    raise ValueError("Some settings files are missing.")

        # Preload data if enabled
        if self.preload_data:
            logging.info("Preloading and validating all data samples...")
            self.initial_graphs = []
            self.final_graphs = []
            self.settings = []
            for idx, (init_file, final_file) in enumerate(zip(self.initial_graph_files, self.final_graph_files)):
                try:
                    initial_data = torch.load(init_file, weights_only=False)
                    final_data = torch.load(final_file, weights_only=False)
                    if self.use_settings and not self.identical_settings:
                        setting = torch.load(self.settings_files[idx], weights_only=False)
                    elif self.use_settings and self.identical_settings:
                        setting = self.settings
                    else:
                        setting = None

                    # **Step 1: Verify Initial Node Feature Dimension**
                    if initial_data.x.shape[1] != self.expected_initial_x_dim:
                        logging.error(f"Sample {idx}: Expected initial node feature dimension {self.expected_initial_x_dim}, "
                                      f"got {initial_data.x.shape[1]}.")
                        raise ValueError(f"Sample {idx}: Inconsistent initial node feature dimensions.")

                    # **Step 2: Verify Settings Dimension**
                    if self.use_settings:
                        if setting is None:
                            logging.error(f"Sample {idx}: Settings are enabled but settings data is missing.")
                            raise ValueError(f"Sample {idx}: Missing settings data.")
                        if setting.shape[0] != self.expected_settings_dim:
                            logging.error(f"Sample {idx}: Expected settings dimension {self.expected_settings_dim}, "
                                          f"got {setting.shape[0]}.")
                            raise ValueError(f"Sample {idx}: Inconsistent settings dimensions.")

                    # **Step 3: Concatenate Settings to Node Features**
                    if self.use_settings:
                        num_nodes = initial_data.num_nodes
                        settings_expanded = setting.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, D_settings]
                        initial_data.x = torch.cat([initial_data.x, settings_expanded], dim=1)  # Shape: [num_nodes, D_total]

                        # **Step 4: Verify Total Node Feature Dimension After Concatenation**
                        if initial_data.x.shape[1] != self.expected_total_x_dim:
                            logging.error(f"Sample {idx}: Expected total node feature dimension {self.expected_total_x_dim}, "
                                          f"got {initial_data.x.shape[1]}.")
                            raise ValueError(f"Sample {idx}: Inconsistent total node feature dimensions after concatenation.")

                    # **Step 5: Extract Positions**
                    initial_data.pos = initial_data.x[:, :3]  # Assuming first 3 features are x, y, z

                    # **Step 6: Compute Edge Attributes if Required**
                    if self.use_edge_attr:
                        self._compute_edge_attr(initial_data, idx)
                    else:
                        initial_data.edge_attr = None  # Explicitly set to None if not used
                        logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

                    # **Step 7: Assign Target Node Features Based on Task**
                    if self.task == 'predict_n6d':
                        initial_data.y = final_data.x[:, :6]  # Shape: [num_nodes, 6]
                    elif self.task == 'predict_n4d':
                        initial_data.y = final_data.x[:, :4]  # Shape: [num_nodes, 4]
                    elif self.task == 'predict_n2d':
                        initial_data.y = final_data.x[:, :2]  # Shape: [num_nodes, 2]
                    else:
                        raise ValueError(f"Unknown task: {self.task}")

                    # **Step 8: Initialize 'batch' Attribute if Missing**
                    if not hasattr(initial_data, 'batch') or initial_data.batch is None:
                        initial_data.batch = torch.zeros(initial_data.num_nodes, dtype=torch.long)
                        logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

                    # Append preloaded and validated data
                    self.initial_graphs.append(initial_data)
                    self.final_graphs.append(final_data)
                    if self.use_settings and not self.identical_settings:
                        self.settings.append(setting)

                except Exception as e:
                    logging.error(f"Error processing sample {idx}: {e}")
                    raise e

            logging.info(f"Preloaded and validated {len(self.initial_graphs)} samples successfully.")
        else:
            self.initial_graphs = None
            self.final_graphs = None
            if self.use_settings and not self.identical_settings:
                self.settings = None

        logging.info(f"Initialized StepPairGraphDataset with {len(self)} samples. "
                     f"Preload data: {self.preload_data}, Use edge_attr: {self.use_edge_attr}, "
                     f"Edge attr method: {self.edge_attr_method}")

    def __len__(self):
        return len(self.initial_graph_files)

    def __getitem__(self, idx):
        if self.preload_data:
            initial_graph = self.initial_graphs[idx]
            final_graph = self.final_graphs[idx]
            if self.use_settings and not self.identical_settings:
                setting = self.settings[idx]
            elif self.use_settings and self.identical_settings:
                setting = self.settings
            else:
                setting = None
        else:
            initial_graph = torch.load(self.initial_graph_files[idx], weights_only=False)
            final_graph = torch.load(self.final_graph_files[idx], weights_only=False)
            if self.use_settings and not self.identical_settings:
                setting = torch.load(self.settings_files[idx], weights_only=False)
            elif self.use_settings and self.identical_settings:
                setting = self.settings
            else:
                setting = None

            # **Step 1: Verify Individual Feature Dimensions Before Concatenation**
            # Verify initial node feature dimension
            if initial_graph.x.shape[1] != self.expected_initial_x_dim:
                logging.error(f"Sample {idx}: Expected initial node feature dimension {self.expected_initial_x_dim}, "
                              f"got {initial_graph.x.shape[1]}.")
                raise ValueError(f"Sample {idx}: Inconsistent initial node feature dimensions.")

            # Verify settings feature dimension
            if self.use_settings:
                if setting is None:
                    logging.error(f"Sample {idx}: Settings are enabled but settings data is missing.")
                    raise ValueError(f"Sample {idx}: Missing settings data.")
                if setting.shape[0] != self.expected_settings_dim:
                    logging.error(f"Sample {idx}: Expected settings dimension {self.expected_settings_dim}, "
                                  f"got {setting.shape[0]}.")
                    raise ValueError(f"Sample {idx}: Inconsistent settings dimensions.")

            # **Step 2: Concatenate Settings to Node Features (If Applicable)**
            if self.use_settings:
                num_nodes = initial_graph.num_nodes
                settings_expanded = setting.unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, D_settings]
                initial_graph.x = torch.cat([initial_graph.x, settings_expanded], dim=1)  # Shape: [num_nodes, D_total]

                # **Step 3: Verify Total Node Feature Dimension After Concatenation**
                if initial_graph.x.shape[1] != self.expected_total_x_dim:
                    logging.error(f"Sample {idx}: Expected total node feature dimension {self.expected_total_x_dim}, "
                                  f"got {initial_graph.x.shape[1]}.")
                    raise ValueError(f"Sample {idx}: Inconsistent total node feature dimensions after concatenation.")

            # **Optional: Log shapes for the first few samples**
            if idx < 5 and self.use_settings:  # Adjust the number as needed
                logging.debug(f"Sample {idx}: Node features shape after concatenation: {initial_graph.x.shape}")

            # **Step 4: Extract Positions**
            initial_graph.pos = initial_graph.x[:, :3]  # Assuming first 3 features are x, y, z coordinates

            # **Step 5: Compute Edge Attributes if Required**
            if self.use_edge_attr:
                self._compute_edge_attr(initial_graph, idx)
            else:
                initial_graph.edge_attr = None  # Explicitly set to None if not used
                logging.debug(f"Sample {idx}: edge_attr not computed (use_edge_attr=False)")

            # **Step 6: Assign Target Node Features Based on Task**
            if self.task == 'predict_n6d':
                initial_graph.y = final_graph.x[:, :6]  # Shape: [num_nodes, 6]
            elif self.task == 'predict_n4d':
                initial_graph.y = final_graph.x[:, :4]  # Shape: [num_nodes, 4]
            elif self.task == 'predict_n2d':
                initial_graph.y = final_graph.x[:, :2]  # Shape: [num_nodes, 2]
            else:
                raise ValueError(f"Unknown task: {self.task}")

            # **Step 7: Initialize 'batch' Attribute if Missing**
            if not hasattr(initial_graph, 'batch') or initial_graph.batch is None:
                initial_graph.batch = torch.zeros(initial_graph.num_nodes, dtype=torch.long)
                logging.debug(f"Sample {idx}: Initialized 'batch' attribute with zeros.")

        return initial_graph

    def _compute_edge_attr(self, graph, idx):
        """
        Computes and assigns edge attributes based on the selected method.

        Args:
            graph (torch_geometric.data.Data): The graph data object.
            idx (int): Index of the data sample.
        """
        row, col = graph.edge_index
        pos_diff = graph.pos[row] - graph.pos[col]  # Shape: [num_edges, 3]

        if self.edge_attr_method == "v0":
            # Use pos_diff only, with standardization
            edge_attr = pos_diff  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v0n":
            # Use normalized pos_diff, without standardization
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            edge_attr = pos_diff / norm  # Shape: [num_edges, 3]
        elif self.edge_attr_method == "v1":
            # Use pos_diff + distance, all standardized
            distance = torch.norm(pos_diff, p=2, dim=1, keepdim=True)  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v1n":
            # Use normalized pos_diff + distance, only distance standardized
            norm = torch.norm(pos_diff, p=2, dim=1, keepdim=True).clamp(min=1e-10)
            normalized_pos_diff = pos_diff / norm  # Shape: [num_edges, 3]
            edge_attr = torch.cat([normalized_pos_diff, norm], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2":
            # Use pos_diff + 1/squared distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v2n":
            # Use normalized pos_diff + 1/squared distance, only 1/squared distance standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            normalized_pos_diff = pos_diff / torch.sqrt(squared_distance)  # Shape: [num_edges, 3]
            inverse_squared_distance = 1 / squared_distance  # Shape: [num_edges, 1]
            edge_attr = torch.cat([normalized_pos_diff, inverse_squared_distance], dim=1)  # Shape: [num_edges, 4]
        elif self.edge_attr_method == "v3":
            # Use pos_diff / squared_distance, all standardized
            squared_distance = (torch.norm(pos_diff, p=2, dim=1, keepdim=True) ** 2).clamp(min=1e-10)  # Shape: [num_edges, 1]
            edge_attr = pos_diff / squared_distance  # Shape: [num_edges, 3]
        else:
            raise ValueError(f"Invalid edge_attr_method: {self.edge_attr_method}")

        # Standardization logic
        if self.edge_attr_method in ["v0", "v1", "v2", "v3"]:
            # Standardize all features
            eps = 1e-10
            edge_attr_mean = edge_attr.mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr.std(dim=0, keepdim=True)
            edge_attr = (edge_attr - edge_attr_mean) / (edge_attr_std + eps)  # Shape: [num_edges, ...]
        elif self.edge_attr_method in ["v1n", "v2n"]:
            # Standardize only the magnitude-related feature (last column)
            eps = 1e-10
            magnitude_indices = [-1]  # Last column contains the magnitude feature
            edge_attr_mean = edge_attr[:, magnitude_indices].mean(dim=0, keepdim=True)
            edge_attr_std = edge_attr[:, magnitude_indices].std(dim=0, keepdim=True)
            edge_attr[:, magnitude_indices] = (edge_attr[:, magnitude_indices] - edge_attr_mean) / (edge_attr_std + eps)

        graph.edge_attr = edge_attr
        logging.debug(f"Sample {idx}: Computed edge_attr with shape {graph.edge_attr.shape} using method {self.edge_attr_method}")


class GraphDataLoaders:
    def __init__(
        self,
        initial_graph_dir,
        final_graph_dir,
        settings_dir,
        task='predict_n6d',
        use_edge_attr=False,
        edge_attr_method="v0",
        preload_data=False,
        batch_size=32,
        n_train=1000,
        n_val=200,
        n_test=200,
        expected_initial_x_dim=6,
        expected_settings_dim=6
    ):
        """
        Initializes the GraphDataLoaders.

        Args:
            initial_graph_dir (str): Directory containing initial graph .pt files.
            final_graph_dir (str): Directory containing final graph .pt files.
            settings_dir (str): Directory containing settings .pt files.
            task (str, optional): Prediction task. Defaults to 'predict_n6d'.
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
                                            Defaults to False.
            edge_attr_method (str, optional): Method for edge attribute computation. Defaults to 'v0'.
            preload_data (bool, optional): If True, preloads all data into memory. Defaults to False.
            batch_size (int, optional): Batch size for the DataLoaders. Defaults to 32.
            n_train (int, optional): Number of training samples. Defaults to 1000.
            n_val (int, optional): Number of validation samples. Defaults to 200.
            n_test (int, optional): Number of testing samples. Defaults to 200.
            expected_initial_x_dim (int, optional): Expected dimension of initial node features. Defaults to 6.
            expected_settings_dim (int, optional): Expected dimension of settings features. Defaults to 6.
        """
        # Initialize the dataset with new parameters
        self.dataset = GraphDataset(
            initial_graph_dir=initial_graph_dir,
            final_graph_dir=final_graph_dir,
            settings_dir=settings_dir,
            task=task,
            use_edge_attr=use_edge_attr,
            edge_attr_method=edge_attr_method,
            preload_data=preload_data,
            expected_initial_x_dim=expected_initial_x_dim,
            expected_settings_dim=expected_settings_dim
        )

        # Sort the dataset indices based on the integer extracted from filenames
        sorted_indices = sorted(
            range(len(self.dataset)),
            key=lambda idx: int(re.search(r'graph_(\d+)', self.dataset.initial_files[idx]).group(1))
        )

        # Total samples required
        total_samples = len(self.dataset)
        n_total = n_train + n_val + n_test

        if n_total > total_samples:
            raise ValueError(f"n_train + n_val + n_test ({n_total}) exceeds total dataset size ({total_samples}).")

        # Select test indices as the last n_test samples
        test_indices = sorted_indices[-n_test:]
        remaining_indices = sorted_indices[:-n_test]

        # Split remaining_indices into train and val
        n_remaining = len(remaining_indices)
        if n_train + n_val > n_remaining:
            raise ValueError(f"n_train + n_val ({n_train + n_val}) exceeds the remaining dataset size ({n_remaining}) after excluding test samples.")

        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:n_train + n_val]

        # Create subsets
        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)

        self.batch_size = batch_size

        # Initialize DataLoaders as None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._all_data_loader = None

        logging.info(f"Initialized GraphDataLoaders with {n_train} train, {n_val} val, and {n_test} test samples.")

    def get_train_loader(self, collate_fn=None):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,  # Shuffle only the training data
                collate_fn=collate_fn
            )
        return self._train_loader

    def get_val_loader(self, collate_fn=None):
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        return self._val_loader

    def get_test_loader(self, collate_fn=None):
        if self._test_loader is None:
            self._test_loader = DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        return self._test_loader

    def get_all_data_loader(self, collate_fn=None):
        """
        Returns a DataLoader for the entire dataset as a single batch, without splitting.
        """
        if self._all_data_loader is None:
            self._all_data_loader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                shuffle=False,
                collate_fn=collate_fn
            )
        return self._all_data_loader


class StepPairGraphDataLoaders(GraphDataLoaders):
    def __init__(
        self,
        graph_data_dir,
        initial_step,
        final_step,
        task='predict_n6d',
        use_settings=False,
        identical_settings=False,
        settings_file=None,
        use_edge_attr=False,
        edge_attr_method="v0",
        preload_data=False,
        batch_size=32,
        n_train=1000,
        n_val=200,
        n_test=200,
        expected_initial_x_dim=6,
        expected_settings_dim=6
    ):
        """
        Initializes the StepPairGraphDataLoaders.

        Args:
            graph_data_dir (str): Base directory containing the graph data organized by sequence steps.
            initial_step (int): Index of the initial sequence step.
            final_step (int): Index of the final sequence step.
            task (str, optional): Prediction task. Defaults to 'predict_n6d'.
            use_settings (bool, optional): Flag indicating whether to use settings.
            identical_settings (bool, optional): Whether settings are identical across samples.
            settings_file (str, optional): Path to the settings file (used if identical_settings is True).
            use_edge_attr (bool, optional): Flag indicating whether to compute edge attributes.
            edge_attr_method (str, optional): Method for edge attribute computation.
            preload_data (bool, optional): If True, preloads all data into memory.
            batch_size (int, optional): Batch size for the DataLoaders. Defaults to 32.
            n_train (int, optional): Number of training samples. Defaults to 1000.
            n_val (int, optional): Number of validation samples. Defaults to 200.
            n_test (int, optional): Number of testing samples. Defaults to 200.
            expected_initial_x_dim (int, optional): Expected dimension of initial node features. Defaults to 6.
            expected_settings_dim (int, optional): Expected dimension of settings features. Defaults to 6.
        """
        # Initialize the dataset with new parameters
        self.dataset = StepPairGraphDataset(
            graph_data_dir=graph_data_dir,
            initial_step=initial_step,
            final_step=final_step,
            task=task,
            use_settings=use_settings,
            identical_settings=identical_settings,
            settings_file=settings_file,
            use_edge_attr=use_edge_attr,
            edge_attr_method=edge_attr_method,
            preload_data=preload_data,
            subsample_size=None,  # Handle subsampling manually
            expected_initial_x_dim=expected_initial_x_dim,
            expected_settings_dim=expected_settings_dim
        )

        # Sort the dataset indices based on the integer extracted from filenames
        sorted_indices = sorted(
            range(len(self.dataset)),
            key=lambda idx: int(re.search(r'graph_(\d+)', os.path.basename(self.dataset.initial_graph_files[idx])).group(1))
        )

        # Total samples required
        total_samples = len(self.dataset)
        n_total = n_train + n_val + n_test

        if n_total > total_samples:
            raise ValueError(f"n_train + n_val + n_test ({n_total}) exceeds total dataset size ({total_samples}).")

        # Select test indices as the last n_test samples
        test_indices = sorted_indices[-n_test:]
        remaining_indices = sorted_indices[:-n_test]

        # Split remaining_indices into train and val
        n_remaining = len(remaining_indices)
        if n_train + n_val > n_remaining:
            raise ValueError(f"n_train + n_val ({n_train + n_val}) exceeds the remaining dataset size ({n_remaining}) after excluding test samples.")

        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:n_train + n_val]

        # Create subsets
        self.train_set = Subset(self.dataset, train_indices)
        self.val_set = Subset(self.dataset, val_indices)
        self.test_set = Subset(self.dataset, test_indices)

        self.batch_size = batch_size

        # Initialize DataLoaders as None
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self._all_data_loader = None

        logging.info(f"Initialized StepPairGraphDataLoaders with {n_train} train, {n_val} val, and {n_test} test samples.")


def verify_dataset_consistency(dataset, expected_total_x_dim, expected_edge_attr_dim=None):
    """
    Verifies that all samples in the dataset have consistent feature dimensions.

    Args:
        dataset (Dataset): The dataset to verify.
        expected_total_x_dim (int): The expected node feature dimension.
        expected_edge_attr_dim (int, optional): The expected edge_attr dimension. Defaults to None.
    """
    logging.info("Verifying dataset consistency...")
    for idx in range(len(dataset)):
        try:
            data = dataset[idx]
            if data.x.shape[1] != expected_total_x_dim:
                logging.error(f"Sample {idx}: Expected node feature dimension {expected_total_x_dim}, got {data.x.shape[1]}.")
                raise ValueError(f"Sample {idx}: Inconsistent node feature dimensions.")
            if expected_edge_attr_dim is not None:
                if data.edge_attr is None:
                    logging.error(f"Sample {idx}: Missing edge_attr.")
                    raise ValueError(f"Sample {idx}: Missing edge_attr.")
                if data.edge_attr.shape[1] != expected_edge_attr_dim:
                    logging.error(f"Sample {idx}: Expected edge_attr dimension {expected_edge_attr_dim}, got {data.edge_attr.shape[1]}.")
                    raise ValueError(f"Sample {idx}: Inconsistent edge_attr dimensions.")
        except Exception as e:
            logging.error(f"Verification failed for sample {idx}: {e}")
            raise e
    logging.info("Dataset consistency verification passed.")
