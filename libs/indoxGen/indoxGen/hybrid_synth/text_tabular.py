import pandas as pd
from typing import List, Dict, Any
import warnings

from .llm_generator import TextDataGenerator

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import handling for TabularGANConfig and TabularGANTrainer
try:
    from indoxGen_tensor import TabularGANConfig, TabularGANTrainer

    print("Successfully imported TabularGANConfig and TabularGANTrainer from indoxGen_tensor.")
except ImportError:
    try:
        from indoxGen_torch import TabularGANConfig, TabularGANTrainer

        print("Successfully imported TabularGANConfig and TabularGANTrainer from indoxGen_torch.")
    except ImportError:
        raise ImportError(
            "Neither `indoxGen_tensor` nor `indoxGen_torch` is installed. "
            "Please install one of these packages to proceed."
        )


# Define the LLM initialization function
def initialize_llm_synth(
        generator_llm,
        judge_llm,
        columns: List[str],
        example_data: List[Dict[str, Any]],
        user_instruction: str,
        diversity_threshold: float = 0.5,  # Adjusted for higher diversity
        max_diversity_failures: int = 30,
        verbose: int = 1
):
    """
    Initializes the LLM-based synthetic text generator setup.

    Parameters:
    -----------
    generator_llm : YourLLMGeneratorClass
        The LLM model used to generate synthetic text.
    judge_llm : YourLLMJudgeClass
        The LLM model used to judge the quality of generated text.
    columns : List[str]
        List of text column names to generate.
    example_data : List[Dict[str, Any]]
        A list of example data records for reference during generation.
    user_instruction : str
        Instructions for the LLM on how to generate the text data.
    diversity_threshold : float, optional
        Threshold for diversity of generated text.
    max_diversity_failures : int, optional
        Maximum allowed diversity failures.
    verbose : int, optional
        Verbosity level for logging and output.

    Returns:
    --------
    SyntheticDataGenerator
        Instance of the initialized synthetic data generator.
    """
    return TextDataGenerator(
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        columns=columns,
        example_data=example_data,
        user_instruction=user_instruction,
        diversity_threshold=diversity_threshold,
        max_diversity_failures=max_diversity_failures,
        verbose=verbose
    )


# Define the GAN initialization function
def initialize_gan_synth(
        input_dim: int,
        generator_layers: List[int],
        discriminator_layers: List[int],
        learning_rate: float,
        beta_1: float,
        beta_2: float,
        batch_size: int,
        epochs: int,
        n_critic: int,
        categorical_columns: List[str],
        mixed_columns: Dict[str, Any],
        integer_columns: List[str],
        data: pd.DataFrame,
        device: str = 'cpu'
):
    """
    Initializes the GAN setup for generating numerical data.

    Parameters:
    -----------
    input_dim : int
        Dimension of the input data.
    generator_layers : List[int]
        Sizes of layers in the generator network.
    discriminator_layers : List[int]
        Sizes of layers in the discriminator network.
    learning_rate : float
        Learning rate for training the GAN.
    beta_1 : float
        Beta1 hyperparameter for the Adam optimizer.
    beta_2 : float
        Beta2 hyperparameter for the Adam optimizer.
    batch_size : int
        Batch size for GAN training.
    epochs : int
        Number of epochs to train the GAN.
    n_critic : int
        Number of discriminator updates per generator update.
    categorical_columns : List[str]
        List of categorical columns (if any) in the numerical data.
    mixed_columns : Dict[str, Any]
        Dictionary of mixed column types (if any).
    integer_columns : List[str]
        List of integer columns in the numerical data.
    data : pd.DataFrame
        The dataset containing numerical columns to train the GAN.
    device : str, optional
        Device to run the GAN ('cpu' or 'cuda').

    Returns:
    --------
    TabularGANTrainer
        Instance of the initialized GAN setup.
    """
    trainer = TabularGANTrainer(
        config=TabularGANConfig(
            input_dim=input_dim,
            generator_layers=generator_layers,
            discriminator_layers=discriminator_layers,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            batch_size=batch_size,
            epochs=epochs,
            n_critic=n_critic
        ),
        categorical_columns=categorical_columns,
        mixed_columns=mixed_columns,
        integer_columns=integer_columns,
        device=device
    )
    trainer.train(data, patience=15, verbose=1)
    return trainer


# Define the main pipeline class that integrates both the LLM and GAN setups
class TextTabularSynth:
    """
    A class to generate synthetic data combining GAN for numerical data
    and LLM for text data.
    """

    def __init__(self, tabular: TabularGANTrainer, text: TextDataGenerator):
        """
        Initializes the TextTabularSynth pipeline.

        Parameters:
        -----------
        tabular : TabularGANTrainer
            Instance of the initialized GAN trainer for numerical data.
        text : TextDataGenerator
            Instance of the initialized synthetic text generator.
        """
        self.tabular = tabular
        self.text = text

    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data for both numerical and text columns.

        Parameters:
        -----------
        num_samples : int
            Number of synthetic data samples to generate.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the combined synthetic data.
        """
        # 1. Generate synthetic numerical data using the GAN
        synthetic_numerical_data = self.tabular.generate_samples(num_samples)

        # 2. Generate synthetic text data using the LLM with numerical context
        synthetic_text_data_list = []
        for idx in range(num_samples):
            # Extract numerical context from the generated numerical data
            numerical_context = synthetic_numerical_data.iloc[idx].to_dict()

            # Generate corresponding text data based on the numerical context
            generated_text = self.text._generate_single_data_point(context=numerical_context)

            if generated_text and all(col in generated_text for col in self.text.columns):
                synthetic_text_data_list.append(generated_text)
            else:
                # Handle cases where text generation fails
                synthetic_text_data_list.append({col: '' for col in self.text.columns})

        # Convert the list of generated text data to a DataFrame
        synthetic_text_data = pd.DataFrame(synthetic_text_data_list)

        # 3. Combine the numerical and text data into a single DataFrame
        synthetic_numerical_data.reset_index(drop=True, inplace=True)
        synthetic_text_data.reset_index(drop=True, inplace=True)
        synthetic_data = pd.concat([synthetic_numerical_data, synthetic_text_data], axis=1)

        return synthetic_data
